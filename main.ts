// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry serves the OpenAI Chat Completions API and the Anthropic Messages API, each in JSON and
// in server-sent events, over ten simulated free-tier providers on three wires (openai, anthropic,
// gemini). For every request it picks the best free model that meets the request's requirements, by
// the brief's quality table, and fails over to the next best when a provider refuses (402, 429, 5xx
// or unreachable). Streams are buffered whole and re-emitted in the client's canonical event order,
// so a translated stream always carries the provider's text whole and its usage counts exactly.
//
// ROUTING
//   Candidates are ranked by rating_milli descending. A provider is demoted below every rating
//   gap only after two refusals in a row (an outage, or its budget spent): one refusal is usually
//   a single call on its failure schedule, and the provider serves the next request. Among equal
//   ratings the provider called least so far wins. A provider is never dropped for refusals alone:
//   a scheduled failure is one call, and the same provider serves the next request.
//   JSON-mode requests from an OpenAI client never go to an Anthropic-wire provider: that wire
//   cannot carry response_format, so the provider would echo json_mode absent and the request would
//   fail its wire check. Gemini carries it as responseMimeType, OpenAI as response_format.
//
// WIRING
//   Every translation preserves what the provider echoes and the client compares: the system prompt,
//   the user text (images included), the tool names in order, the stop sequences in order,
//   max_tokens, and JSON mode. The provider's answer text is always "ANSWER ECHO{...}" whole; the
//   reply is built from that text and the provider's usage, never invented.
//
// DETERMINISM
//   Every choice is a function of the brief, the requests and the providers' answers so far: no
//   wall clock, no unseeded randomness, one request at a time, one provider call at a time. The
//   poll loop's timing never reaches a request or a reply.
//
//   `node main.ts --list` prints the catalog document and exits 0. Run outside a trial, with no
//   MAKEFUN_PROMPT_FILE and no MAKEFUN_ENV_URL set, it prints {"providers":[]}: it has no brief to
//   read and no provider route to probe. The register body's catalog is that same document, byte
//   for byte.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routing, the routes, register, poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the client and upstream wires: request and answer translation, SSE, usage.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type Wire,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicStreamText,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIAnswer,
  answerTextOf,
  computeUsage,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIModelList,
  openAIStreamText,
  openAIRequestToAnthropic,
  openAIToGemini,
  parseAnthropicStream,
  parseGeminiStream,
  parseOpenAIStream,
  usageOfAnswer,
  type ParsedStream,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** A provider that refused on either of the last two requests loses this many rating milli. */
const RECENT_FAILURE_PENALTY = 200;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Inference calls made to each provider so far, successes and refusals alike. */
const callsByProvider = new Map<string, number>();
/** Refusals in a row per provider: a success resets it. Two in a row means the provider is dead
 *  for now (an outage, or its budget spent), not one call on its failure schedule. */
const consecutiveFailures = new Map<string, number>();
/** The last request number through which each provider is demoted, after two refusals in a row. */
const demoteThrough = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, each id once: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best first. The ranking is the quality table's
 * rating, with a provider demoted below every rating gap only after two refusals in a row (an
 * outage, or its budget spent): one refusal is usually a single call on its failure schedule, and
 * the provider serves the next request. Equal ratings break by the fewest calls so far, then
 * provider and model id. A JSON-mode OpenAI request never takes an Anthropic-wire provider: that
 * wire cannot carry response_format, so the provider would echo json_mode absent.
 */
function candidatesFor(requires: Requirement, jsonWire: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonWire && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    const penalty = (c: Candidate): number => ((demoteThrough.get(c.probe.provider.id) ?? 0) >= manifestIndex ? RECENT_FAILURE_PENALTY : 0);
    const scoreA = a.rating - penalty(a);
    const scoreB = b.rating - penalty(b);
    if (scoreA !== scoreB) return scoreB - scoreA;
    const callsA = callsByProvider.get(a.probe.provider.id) ?? 0;
    const callsB = callsByProvider.get(b.probe.provider.id) ?? 0;
    if (callsA !== callsB) return callsA - callsB;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess
  // for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this OpenAI client request asks for JSON mode, which only some wires can carry. */
function jsonWireOf(api: 'openai' | 'anthropic', body: Json): boolean {
  if (api !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream request body for one candidate, in its wire. */
function upstreamBody(api: 'openai' | 'anthropic', body: Json, candidate: Candidate, jsonWire: boolean): Json {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (wire === 'openai') return api === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
  if (wire === 'anthropic') return api === 'anthropic' ? { ...body, model } : openAIRequestToAnthropic(body, model);
  return api === 'openai' ? openAIToGemini(body, jsonWire) : anthropicToGemini(body);
}

/** The upstream route and headers for one candidate. */
function upstreamTarget(candidate: Candidate, stream: boolean): { url: string; headers: Record<string, string> } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'openai') {
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/chat/completions`, headers };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/messages`, headers };
  }
  const path = stream ? 'streamGenerateContent' : 'generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${path}`, headers };
}

interface UpstreamAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

/** Parse one upstream answer, JSON or streamed, into its text and usage. Null when it is not an answer. */
function parseAnswer(wire: Wire, stream: boolean, raw: string): UpstreamAnswer | null {
  if (stream) {
    const parsed: ParsedStream = wire === 'openai' ? parseOpenAIStream(raw) : wire === 'anthropic' ? parseAnthropicStream(raw) : parseGeminiStream(raw);
    if (!parsed.ok) return null;
    return { text: parsed.text, inputTokens: parsed.inputTokens ?? -1, outputTokens: parsed.outputTokens ?? -1 };
  }
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isRecord(value)) return null;
  const text = answerTextOf(wire, value);
  if (text === '') return null;
  const usage = usageOfAnswer(wire, value);
  if (usage === null) return null;
  return { text, inputTokens: usage.input, outputTokens: usage.output };
}

/** One provider call: the answer, or null when the provider refused or could not be reached. */
async function callCandidate(candidate: Candidate, api: 'openai' | 'anthropic', body: Json, jsonWire: boolean): Promise<UpstreamAnswer | null> {
  const stream = body.stream === true;
  const target = upstreamTarget(candidate, stream);
  const upstream = upstreamBody(api, body, candidate, jsonWire);
  callsByProvider.set(candidate.probe.provider.id, (callsByProvider.get(candidate.probe.provider.id) ?? 0) + 1);
  let res: Response;
  try {
    res = await fetch(target.url, {
      method: 'POST',
      headers: target.headers,
      body: JSON.stringify(upstream),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
  if (!res.ok) return null;
  const raw = await res.text();
  const answer = parseAnswer(candidate.probe.provider.wire, stream, raw);
  if (answer === null) return null;
  if (!answer.text.startsWith('ANSWER ECHO')) return null;
  return answer;
}

/**
 * Record that a provider refused a call on this request. One refusal is a scheduled failure and
 * changes nothing; the second in a row demotes the provider for the next two requests, so a dead
 * provider is not retried first on every request while it is out.
 */
function noteFailure(providerId: string): void {
  const consecutive = (consecutiveFailures.get(providerId) ?? 0) + 1;
  consecutiveFailures.set(providerId, consecutive);
  if (consecutive >= 2) demoteThrough.set(providerId, manifestIndex + 2);
}

/** Record that a provider answered a call: its refusal streak ends. */
function noteSuccess(providerId: string): void {
  consecutiveFailures.set(providerId, 0);
}

/**
 * Serve one client request: try the ranked candidates in order until one answers, then translate
 * the answer into the client's wire. A refusal is never the answer: the next candidate serves it.
 */
async function serve(api: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  const jsonWire = jsonWireOf(api, body);
  const candidates = candidatesFor(requires, jsonWire);
  if (candidates.length === 0) {
    return sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  const tried = new Set<string>();
  for (const candidate of candidates) {
    const key = `${candidate.probe.provider.id}/${candidate.model.id}`;
    if (tried.has(key)) continue;
    tried.add(key);
    const answer = await callCandidate(candidate, api, body, jsonWire);
    if (answer === null) {
      noteFailure(candidate.probe.provider.id);
      continue;
    }
    noteSuccess(candidate.probe.provider.id);
    const usage =
      answer.inputTokens >= 0 && answer.outputTokens >= 0
        ? { input: answer.inputTokens, output: answer.outputTokens }
        : computeUsage(api, body, answer.text);
    const model = candidate.model.id;
    const wire = candidate.probe.provider.wire;
    const stream = body.stream === true;
    if (stream) {
      const text = api === 'openai' ? openAIStreamText(model, answer.text, usage.input, usage.output) : anthropicStreamText(model, answer.text, usage.input, usage.output);
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(text);
      return;
    }
    if (api === 'openai') {
      if (wire === 'openai') {
        // The provider's own answer is already the shape the client wants; report the usage the
        // provider recorded, which the stream formula also gives.
        sendJson(res, 200, {
          id: 'chatcmpl-sim',
          object: 'chat.completion',
          created: 1767225600,
          model,
          choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
          usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
        });
      } else if (wire === 'anthropic') {
        sendJson(res, 200, anthropicToOpenAIAnswer({ content: [{ type: 'text', text: answer.text }], stop_reason: 'end_turn', usage: { input_tokens: usage.input, output_tokens: usage.output } }, model));
      } else {
        sendJson(res, 200, geminiToOpenAI({ candidates: [{ content: { parts: [{ text: answer.text }], role: 'model' }, finishReason: 'STOP', index: 0 }], usageMetadata: { promptTokenCount: usage.input, candidatesTokenCount: usage.output, totalTokenCount: usage.input + usage.output }, modelVersion: model }, model));
      }
      return;
    }
    if (wire === 'anthropic') {
      sendJson(res, 200, {
        id: 'msg_sim',
        type: 'message',
        role: 'assistant',
        model,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: usage.output },
      });
    } else if (wire === 'openai') {
      sendJson(res, 200, openAIToAnthropicShim(answer.text, model, usage));
    } else {
      sendJson(res, 200, geminiToAnthropic({ candidates: [{ content: { parts: [{ text: answer.text }], role: 'model' }, finishReason: 'STOP', index: 0 }], usageMetadata: { promptTokenCount: usage.input, candidatesTokenCount: usage.output, totalTokenCount: usage.input + usage.output }, modelVersion: model }, model));
    }
    return;
  }
  sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

/** An OpenAI-shaped answer text as an Anthropic Messages object (the provider's own shape rebuilt). */
function openAIToAnthropicShim(text: string, model: string, usage: { input: number; output: number }): Json {
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
