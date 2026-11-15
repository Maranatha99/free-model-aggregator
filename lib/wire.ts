// The client wires and the three upstream wires this entry speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. Each request is answered by
// one of ten providers on its own wire: openai, anthropic or gemini. This file translates both ways:
//
//   requests:  openai <-> anthropic, openai -> gemini, anthropic -> gemini
//   answers:   every upstream wire -> either client wire, JSON and streaming
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body
// and the same provider answer give the same bytes on a replay. Streams are buffered whole and then
// re-emitted in the client's canonical event order, so a translated stream always carries the
// provider's text whole and its usage counts exactly (the usage the stream itself carried, or the
// counts the env's own formula gives for the request and the answer text, which are the provider's).
// Model listings use fixed timestamps for the same reason: a `created` that moved between runs would
// be a nondeterministic document.
//
// What the translations must preserve, because the provider echoes what it parsed and the client
// compares that echo with what it sent: the system prompt, the user text (images included), the tool
// names in order, the stop sequences in order, max_tokens, and JSON mode. A field rewritten or
// dropped loses the request.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
export const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
export const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of the last user message of a messages array, as a provider's parser reads it. */
function lastUserText(messages: unknown): string {
  if (!Array.isArray(messages)) return '';
  let text = '';
  for (const raw of messages) {
    if (!isRecord(raw) || raw.role !== 'user') continue;
    text = textOf(raw.content);
  }
  return text;
}

/** The system text of an OpenAI request: its system messages joined, as a provider's parser reads it. */
function openAISystemText(body: Json): string {
  if (!Array.isArray(body.messages)) return '';
  return body.messages
    .flatMap((raw) => (isRecord(raw) && raw.role === 'system' ? [textOf(raw.content)] : []))
    .filter((t) => t !== '')
    .join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One OpenAI content part as an Anthropic content block: text or an image; null for the rest. */
function blockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: system
 * messages become the system prompt, image_url parts become image blocks, tool messages become
 * tool_result blocks, tools and tool_choice are rewritten, and the output cap, temperature, top_p,
 * stop sequences and streaming pass through. JSON mode has no Anthropic spelling: an OpenAI request
 * that carries response_format must never be sent to an Anthropic-wire provider.
 */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const system = openAISystemText(body);
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) }],
      });
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = blockOf(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'auto' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object. */
export function anthropicToOpenAIAnswer(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = textOf(content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `chatcmpl_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: finishReasonOf(body.stop_reason), logprobs: null }],
    usage: { prompt_tokens: numberOrZero(usage.input_tokens), completion_tokens: numberOrZero(usage.output_tokens), total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens) },
  };
}

/** Split a data: URL into its media type and base64 payload, or null when it is a plain URL. */
function splitDataUrl(url: string): { media: string; data: string } | null {
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data === null) return null;
  return { media: data[1] !== '' ? data[1] : 'image/png', data: data[2] };
}

/** One image as a Gemini part: inlineData for a data URL or base64 source, fileData for a plain URL. */
function geminiImagePart(url: string | null, base64: string | null, media: string): Json | null {
  if (base64 !== null) return { inlineData: { mimeType: media, data: base64 } };
  if (url !== null) return { fileData: { fileUri: url } };
  return null;
}

/** The image parts of one message's content, in OpenAI, Anthropic or Gemini spelling. */
function imagePartsOf(content: unknown): Json[] {
  const out: Json[] = [];
  if (!Array.isArray(content)) return out;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const split = splitDataUrl(block.image_url.url);
      const part = split !== null ? geminiImagePart(null, split.data, split.media) : geminiImagePart(block.image_url.url, null, 'image/png');
      if (part !== null) out.push(part);
    }
    if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        const part = geminiImagePart(null, block.source.data, typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png');
        if (part !== null) out.push(part);
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        const part = geminiImagePart(block.source.url, null, 'image/png');
        if (part !== null) out.push(part);
      }
    }
    if (isRecord(block.inlineData) && typeof block.inlineData.data === 'string') {
      out.push({ inlineData: { mimeType: typeof block.inlineData.mimeType === 'string' ? block.inlineData.mimeType : 'image/png', data: block.inlineData.data } });
    }
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request: the system prompt becomes
 * systemInstruction, messages become contents (user text and images, model turns, function
 * responses), tools become functionDeclarations, and max_tokens, temperature, top_p, stop sequences
 * and JSON mode become generationConfig. `jsonMode` must be true only for a client request that
 * carries response_format: a Gemini answer echoes it.
 */
export function openAIToGemini(body: Json, jsonMode: boolean): Json {
  const system = openAISystemText(body);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    if (raw.role === 'tool') {
      contents.push({
        role: 'function',
        parts: [{ functionResponse: { name: '', response: { content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) } } }],
      });
      continue;
    }
    const parts: Json[] = [];
    const text = textOf(raw.content);
    if (text !== '') parts.push({ text });
    for (const part of imagePartsOf(raw.content)) parts.push(part);
    if (raw.role === 'assistant') {
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  if (jsonMode) config.responseMimeType = 'application/json';
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request: the system prompt becomes
 * systemInstruction, messages become contents (user text and images, assistant turns, tool results
 * as function responses), tools become functionDeclarations, and the output cap, temperature, top_p
 * and stop sequences become generationConfig. Anthropic has no JSON mode, so none is set.
 */
export function anthropicToGemini(body: Json): Json {
  const system = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    const text = textOf(content);
    if (raw.role === 'user') {
      if (text !== '') parts.push({ text });
      for (const part of imagePartsOf(content)) parts.push(part);
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_result') continue;
        parts.push({ functionResponse: { name: '', response: { content: toolResultText(block.content) } } });
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    if (text !== '') parts.push({ text });
    for (const block of content) {
      if (!isRecord(block) || block.type !== 'tool_use') continue;
      parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
    }
    if (parts.length > 0) contents.push({ role: 'model', parts });
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The text of a Gemini generateContent answer: its candidates' parts joined. */
function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const parts: string[] = [];
  for (const candidate of candidates) {
    if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) continue;
    for (const part of candidate.content.parts) {
      if (isRecord(part) && typeof part.text === 'string') parts.push(part.text);
    }
  }
  return parts.join('');
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = numberOrZero(usage.promptTokenCount);
  const output = numberOrZero(usage.candidatesTokenCount);
  return {
    id: `chatcmpl_${typeof body.modelVersion === 'string' ? body.modelVersion : 'gemini'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiTextOf(body) }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: `msg_${typeof body.modelVersion === 'string' ? body.modelVersion : 'gemini'}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiTextOf(body) }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

/** The text of a JSON answer on one wire: OpenAI, Anthropic or Gemini. */
export function answerTextOf(wire: 'openai' | 'anthropic' | 'gemini', body: Json): string {
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    return typeof message.content === 'string' ? message.content : textOf(message.content);
  }
  if (wire === 'anthropic') return textOf(body.content);
  return geminiTextOf(body);
}

/** The usage of a JSON answer on one wire, or null when it carries none. */
export function usageOfAnswer(wire: 'openai' | 'anthropic' | 'gemini', body: Json): { input: number; output: number } | null {
  if (wire === 'openai') {
    if (!isRecord(body.usage)) return null;
    return { input: numberOrZero(body.usage.prompt_tokens), output: numberOrZero(body.usage.completion_tokens) };
  }
  if (wire === 'anthropic') {
    if (!isRecord(body.usage)) return null;
    return { input: numberOrZero(body.usage.input_tokens), output: numberOrZero(body.usage.output_tokens) };
  }
  if (!isRecord(body.usageMetadata)) return null;
  return { input: numberOrZero(body.usageMetadata.promptTokenCount), output: numberOrZero(body.usageMetadata.candidatesTokenCount) };
}

/**
 * The usage the env's own formula gives for a client request and an answer text: input is one plus
 * a quarter of the system text, the user text and the tool names the provider parsed; output is one
 * plus a quarter of the answer's length. A faithful translation makes the provider's parse equal
 * the client's, so these counts are the provider's recorded counts.
 */
export function computeUsage(api: 'openai' | 'anthropic', body: Json, answerText: string): { input: number; output: number } {
  let systemText = '';
  let userText = '';
  let tools: string[] = [];
  if (api === 'openai') {
    systemText = openAISystemText(body);
    userText = lastUserText(body.messages);
    if (Array.isArray(body.tools)) {
      tools = body.tools.flatMap((raw) => (isRecord(raw) && isRecord(raw.function) && typeof raw.function.name === 'string' ? [raw.function.name] : []));
    }
  } else {
    systemText = textOf(body.system);
    userText = lastUserText(body.messages);
    if (Array.isArray(body.tools)) {
      tools = body.tools.flatMap((raw) => (isRecord(raw) && typeof raw.name === 'string' ? [raw.name] : []));
    }
  }
  const prompt = systemText.length + userText.length + tools.join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(answerText.length / 4) };
}

/** One server-sent event frame: `event: <name>` when named, then the JSON data line, then a blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The data payloads of an SSE text, in order: the `data:` lines joined per frame. */
function ssePayloads(text: string): string[] {
  const out: string[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) out.push(data.join('\n'));
  }
  return out;
}

/** A parsed upstream stream: the answer text whole and the usage it carried, if any. */
export interface ParsedStream {
  ok: boolean;
  text: string;
  inputTokens: number | null;
  outputTokens: number | null;
}

/** An OpenAI event stream parsed whole: content deltas joined, the finish reason, the usage chunk. */
export function parseOpenAIStream(text: string): ParsedStream {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let sawDone = false;
  for (const payload of ssePayloads(text)) {
    if (payload.trim() === '[DONE]') {
      sawDone = true;
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
  }
  return { ok: sawDone && out !== '', text: out, inputTokens: input, outputTokens: output };
}

/** An Anthropic event stream parsed whole: text deltas joined, the usage in start and delta events. */
export function parseAnthropicStream(text: string): ParsedStream {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let sawStop = false;
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length === 0) continue;
    let value: unknown;
    try {
      value = JSON.parse(data.join('\n'));
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    if (event === 'message_start' && isRecord(value.message) && isRecord(value.message.usage)) {
      input = numberOrZero(value.message.usage.input_tokens);
    }
    if (event === 'content_block_delta' && isRecord(value.delta) && value.delta.type === 'text_delta' && typeof value.delta.text === 'string') {
      out += value.delta.text;
    }
    if (event === 'message_delta' && isRecord(value.usage)) output = numberOrZero(value.usage.output_tokens);
    if (event === 'message_stop') sawStop = true;
  }
  return { ok: sawStop && out !== '', text: out, inputTokens: input, outputTokens: output };
}

/** A Gemini event stream parsed whole: candidate pieces joined, the usage in the final frame. */
export function parseGeminiStream(text: string): ParsedStream {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let sawFinish = false;
  for (const payload of ssePayloads(text)) {
    let value: unknown;
    try {
      value = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    for (const candidate of candidates) {
      if (!isRecord(candidate)) continue;
      if (typeof candidate.finishReason === 'string') sawFinish = true;
      if (!isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) continue;
      for (const part of candidate.content.parts) {
        if (isRecord(part) && typeof part.text === 'string') out += part.text;
      }
    }
    if (isRecord(value.usageMetadata)) {
      input = numberOrZero(value.usageMetadata.promptTokenCount);
      output = numberOrZero(value.usageMetadata.candidatesTokenCount);
    }
  }
  return { ok: sawFinish && out !== '', text: out, inputTokens: input, outputTokens: output };
}

/** Split a text into the provider's 13-character streaming pieces. */
function streamPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/**
 * A whole answer as an OpenAI event stream: a role chunk, the content in pieces, a finish chunk, a
 * usage chunk and [DONE] — the shape the client's stream checker reads.
 */
export function openAIStreamText(model: string, text: string, inputTokens: number, outputTokens: number): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of streamPieces(text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/**
 * A whole answer as an Anthropic event stream: message_start, one text block in pieces, message_delta
 * with the stop reason and the output tokens, then message_stop — the shape the client's checker reads.
 */
export function anthropicStreamText(model: string, text: string, inputTokens: number, outputTokens: number): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_sim', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
    }),
  );
  frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of streamPieces(text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: outputTokens } }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}
