// Unit tests for lib/wire.ts: request and answer translation between the client wires
// (OpenAI, Anthropic) and the upstream wires (OpenAI, Anthropic, Gemini), the stream
// parsers and emitters, and the usage formula the env uses to score a response.
//
// Run: node --test test/
// The trial's own checks are the authority; these tests pin the translations that feed them.

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  anthropicStreamText,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIAnswer,
  computeUsage,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIRequestToAnthropic,
  openAIStreamText,
  openAIToAnthropic,
  openAIToGemini,
  parseAnthropicStream,
  parseGeminiStream,
  parseOpenAIStream,
} from '../lib/wire.ts';

const IMG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

const openAIBody = {
  model: 'auto',
  messages: [
    { role: 'system', content: 'Be careful.' },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Read the image and call lookup_customer.' },
        { type: 'image_url', image_url: { url: `data:image/png;base64,${IMG}` } },
      ],
    },
  ],
  max_tokens: 131,
  temperature: 0.9,
  tools: [{ type: 'function', function: { name: 'lookup_customer', description: 'Look one up.', parameters: { type: 'object', properties: { account_id: { type: 'string' } }, required: ['account_id'] } } }],
  tool_choice: 'auto',
  stop: ['\n\n###'],
  stream: true,
};

const anthropicBody = {
  model: 'auto',
  max_tokens: 448,
  system: 'Be careful.',
  messages: [{ role: 'user', content: 'The billing job failed on account 4407.' }],
  temperature: 0.3,
  tools: [{ name: 'lookup_customer', description: 'Look one up.', input_schema: { type: 'object', properties: { account_id: { type: 'string' } }, required: ['account_id'] } }],
  stop_sequences: ['\n\nHuman:'],
};

test('anthropic request -> openai keeps system, text, tools, stops and stream options', () => {
  const out = anthropicToOpenAI(anthropicBody, 'glm-4.7-flash');
  assert.equal(out.model, 'glm-4.7-flash');
  assert.deepEqual(out.messages, [
    { role: 'system', content: 'Be careful.' },
    { role: 'user', content: 'The billing job failed on account 4407.' },
  ]);
  assert.equal(out.max_tokens, 448);
  assert.deepEqual(out.stop, ['\n\nHuman:']);
  assert.equal(out.tools[0].function.name, 'lookup_customer');
  assert.deepEqual(out.tools[0].function.parameters.properties, { account_id: { type: 'string' } });
});

test('anthropic request -> openai adds stream_options include_usage when streaming', () => {
  const out = anthropicToOpenAI({ ...anthropicBody, stream: true }, 'glm-4.7-flash');
  assert.equal(out.stream, true);
  assert.deepEqual(out.stream_options, { include_usage: true });
});

test('openai request -> anthropic keeps system, image, tools, stops', () => {
  const out = openAIRequestToAnthropic(openAIBody, 'glm-4.7-flash');
  assert.equal(out.model, 'glm-4.7-flash');
  assert.equal(out.system, 'Be careful.');
  assert.equal(out.max_tokens, 131);
  assert.deepEqual(out.stop_sequences, ['\n\n###']);
  const user = out.messages[0];
  assert.equal(user.role, 'user');
  assert.equal(user.content[0].type, 'text');
  assert.equal(user.content[0].text, 'Read the image and call lookup_customer.');
  assert.equal(user.content[1].type, 'image');
  assert.equal(user.content[1].source.type, 'base64');
  assert.equal(user.content[1].source.data, IMG);
  assert.equal(user.content[1].source.media_type, 'image/png');
  assert.equal(out.tools[0].name, 'lookup_customer');
  assert.deepEqual(out.tool_choice, { type: 'auto' });
  assert.equal(out.stream, true);
});

test('openai request -> gemini carries system, image, tools, json mode', () => {
  const out = openAIToGemini({ ...openAIBody, response_format: { type: 'json_object' } }, true);
  assert.deepEqual(out.systemInstruction, { parts: [{ text: 'Be careful.' }] });
  const user = out.contents[0];
  assert.equal(user.role, 'user');
  assert.equal(user.parts[0].text, 'Read the image and call lookup_customer.');
  assert.equal(user.parts[1].inlineData.mimeType, 'image/png');
  assert.equal(user.parts[1].inlineData.data, IMG);
  assert.equal(out.tools[0].functionDeclarations[0].name, 'lookup_customer');
  assert.equal(out.generationConfig.maxOutputTokens, 131);
  assert.deepEqual(out.generationConfig.stopSequences, ['\n\n###']);
  assert.equal(out.generationConfig.responseMimeType, 'application/json');
});

test('openai request -> gemini without json mode sets no responseMimeType', () => {
  const out = openAIToGemini(openAIBody, false);
  assert.equal(out.generationConfig.responseMimeType, undefined);
});

test('anthropic request -> gemini carries system, text, tools, stops', () => {
  const out = anthropicToGemini(anthropicBody);
  assert.deepEqual(out.systemInstruction, { parts: [{ text: 'Be careful.' }] });
  assert.equal(out.contents[0].parts[0].text, 'The billing job failed on account 4407.');
  assert.equal(out.tools[0].functionDeclarations[0].name, 'lookup_customer');
  assert.equal(out.generationConfig.maxOutputTokens, 448);
  assert.deepEqual(out.generationConfig.stopSequences, ['\n\nHuman:']);
  assert.equal(out.generationConfig.responseMimeType, undefined);
});

test('openai answer -> anthropic message keeps text, stop reason and usage', () => {
  const answer = {
    id: 'chatcmpl-sim',
    choices: [{ message: { role: 'assistant', content: 'ANSWER ECHO{}' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 40, completion_tokens: 12 },
  };
  const out = openAIToAnthropic(answer, 'glm-4.7-flash');
  assert.equal(out.type, 'message');
  assert.equal(out.content[0].text, 'ANSWER ECHO{}');
  assert.equal(out.stop_reason, 'end_turn');
  assert.deepEqual(out.usage, { input_tokens: 40, output_tokens: 12 });
});

test('anthropic answer -> openai completion keeps text, stop reason and usage', () => {
  const answer = {
    id: 'msg_sim',
    content: [{ type: 'text', text: 'ANSWER ECHO{}' }],
    stop_reason: 'end_turn',
    usage: { input_tokens: 40, output_tokens: 12 },
  };
  const out = anthropicToOpenAIAnswer(answer, 'glm-4.7-flash');
  assert.equal(out.choices[0].message.content, 'ANSWER ECHO{}');
  assert.equal(out.choices[0].finish_reason, 'stop');
  assert.deepEqual(out.usage, { prompt_tokens: 40, completion_tokens: 12, total_tokens: 52 });
});

test('gemini answer -> both client wires keeps text and usage', () => {
  const answer = {
    candidates: [{ content: { parts: [{ text: 'ANSWER ECHO{}' }], role: 'model' }, finishReason: 'STOP', index: 0 }],
    usageMetadata: { promptTokenCount: 40, candidatesTokenCount: 12, totalTokenCount: 52 },
    modelVersion: 'gemini-3.8-flash',
  };
  const openai = geminiToOpenAI(answer, 'gemini-3.8-flash');
  assert.equal(openai.choices[0].message.content, 'ANSWER ECHO{}');
  assert.equal(openai.choices[0].finish_reason, 'stop');
  assert.deepEqual(openai.usage, { prompt_tokens: 40, completion_tokens: 12, total_tokens: 52 });
  const anthropic = geminiToAnthropic(answer, 'gemini-3.8-flash');
  assert.equal(anthropic.content[0].text, 'ANSWER ECHO{}');
  assert.equal(anthropic.stop_reason, 'end_turn');
  assert.deepEqual(anthropic.usage, { input_tokens: 40, output_tokens: 12 });
});

test('computeUsage matches the env formula: 1 + floor(prompt/4), 1 + floor(text/4)', () => {
  // system "Be careful." (12) + user text (40) + tools "lookup_customer" (15) = 67 -> input 17.
  const usage = computeUsage('openai', openAIBody, 'ANSWER ECHO{"a":1}');
  assert.deepEqual(usage, { input: 1 + Math.floor(67 / 4), output: 1 + Math.floor('ANSWER ECHO{"a":1}'.length / 4) });
  const usageAnthropic = computeUsage('anthropic', anthropicBody, 'ANSWER ECHO{"a":1}');
  // system (12) + user (39) + tools (15) = 66 -> input 17.
  assert.equal(usageAnthropic.input, 1 + Math.floor(66 / 4));
});

test('openai stream round-trips: emit then parse gives the text and usage back', () => {
  const text = 'ANSWER ECHO{"system_present":true}';
  const emitted = openAIStreamText('gpt-oss-120b', text, 18, 9);
  assert.ok(emitted.endsWith('data: [DONE]\n\n'));
  const parsed = parseOpenAIStream(emitted);
  assert.equal(parsed.ok, true);
  assert.equal(parsed.text, text);
  assert.equal(parsed.inputTokens, 18);
  assert.equal(parsed.outputTokens, 9);
});

test('anthropic stream round-trips: emit then parse gives the text and usage back', () => {
  const text = 'ANSWER ECHO{"system_present":true}';
  const emitted = anthropicStreamText('glm-4.7-flash', text, 18, 9);
  assert.ok(emitted.endsWith('event: message_stop\ndata: {"type":"message_stop"}\n\n'));
  const parsed = parseAnthropicStream(emitted);
  assert.equal(parsed.ok, true);
  assert.equal(parsed.text, text);
  assert.equal(parsed.inputTokens, 18);
  assert.equal(parsed.outputTokens, 9);
});

test('gemini stream parses pieces and the final usage frame', () => {
  const text = 'ANSWER ECHO{"system_present":true}';
  const pieces = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  const frames = pieces.map((p) => `data: ${JSON.stringify({ candidates: [{ content: { parts: [{ text: p }], role: 'model' }, index: 0 }] })}\n\n`).join('');
  const last = `data: ${JSON.stringify({ candidates: [{ content: { parts: [{ text: '' }], role: 'model' }, finishReason: 'STOP', index: 0 }], usageMetadata: { promptTokenCount: 18, candidatesTokenCount: 9, totalTokenCount: 27 }, modelVersion: 'gemini-3.8-flash' })}\n\n`;
  const parsed = parseGeminiStream(frames + last);
  assert.equal(parsed.ok, true);
  assert.equal(parsed.text, text);
  assert.equal(parsed.inputTokens, 18);
  assert.equal(parsed.outputTokens, 9);
});
