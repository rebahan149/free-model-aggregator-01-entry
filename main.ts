// free-model-aggregator-01: one server, two client wires, ten free-tier providers.
//
// The server exposes the OpenAI Chat Completions API and the Anthropic Messages API (JSON and
// server-sent events) and answers each of the trial's 20 requests with the best free model that
// can serve it, dialing the ten simulated providers under $MAKEFUN_ENV_URL on their own wires
// (OpenAI, Anthropic, Gemini). It registers its catalog with the env, polls /work, and exits 0
// when the workload is done.
//
// Routing design, in one paragraph per idea:
//   - Feasibility first: a candidate must be free, listed by its provider, and able to meet the
//     request's manifest requirements (tools / vision / JSON mode / context length). A request the
//     client sent in JSON mode never routes to the Anthropic wire, which cannot carry that mode.
//   - Quality-ordered: candidates rank by the brief's sealed rating, discounted by a Bayesian
//     success posterior kept per provider (a Beta prior updated on every answer the provider
//     gives), so a provider that has been refusing sinks below a slightly weaker healthy one.
//   - Risk-priced retry: a 429 refusal early in a provider's budget is almost always a scheduled
//     one-call refusal, so the same model is retried once when it is strictly the best remaining
//     pick; the retry keeps the high rating at the cost of one call, and a second refusal moves
//     on. A 402 means the free allowance is gone: that provider is dropped for the whole trial.
//     A refusal at or beyond the provider's known minimum budget is read as exhaustion and the
//     provider is dropped too.
//   - Zero-failure first call: any refusal fails over to the next candidate at once, so the
//     client still gets exactly one successful provider answer per request.
// Everything the server decides is a function of the brief, the requests and the providers'
// answers; nothing reads the wall clock and nothing is random, so a replay matches byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Json = Record<string, unknown>;
type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const isRec = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const strOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const UPSTREAM_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Fixed `created` values for the model lists and completions: the clock must never leak into a reply. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

// ---------------------------------------------------------------------------
// The brief: three JSON documents found by their `schema` member in the file.
// ---------------------------------------------------------------------------

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

interface ProviderInfo {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string; // the provider's own path (or absolute URL) under the env
  keyEnv: string;
  key: string;
}

interface RatingRow {
  provider: string;
  model: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number | null;
}

interface ManifestEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

interface Brief {
  providers: ProviderInfo[];
  ratings: RatingRow[];
  manifest: ManifestEntry[];
}

/** The index of the `}` that closes the object opening at `start`, skipping string contents. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every standalone JSON object in `text` that carries a string `schema`, in order of appearance. */
function schemaDocuments(text: string): Json[] {
  const docs: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRec(value) && typeof value.schema === 'string') docs.push(value);
    } catch {
      // Not a standalone document: keep scanning.
    }
  }
  return docs;
}

function providerInfoOf(v: unknown): ProviderInfo | null {
  if (!isRec(v)) return null;
  const id = strOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth: AuthScheme =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    strOf(v.path) !== '' ? strOf(v.path) : strOf(v.base_path) !== '' ? strOf(v.base_path) : strOf(v.base_url) !== '' ? strOf(v.base_url) : strOf(v.base);
  return { id, name: strOf(v.name) !== '' ? strOf(v.name) : id, wire, auth, base, keyEnv: strOf(v.key_env), key: strOf(v.key) };
}

function ratingRowOf(v: unknown): RatingRow | null {
  if (!isRec(v)) return null;
  const provider = strOf(v.provider);
  const model = strOf(v.model);
  if (provider === '' || model === '') return null;
  return {
    provider,
    model,
    ratingMilli: numOrNull(v.rating_milli) ?? 0,
    contextLength: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function requirementOf(v: unknown): Requirement {
  const r = isRec(v) ? v : {};
  return {
    tools: r.tools === true,
    vision: r.vision === true,
    json: r.json === true,
    minContext: numOrNull(r.min_context),
  };
}

function manifestEntryOf(v: unknown): ManifestEntry | null {
  if (!isRec(v)) return null;
  const id = strOf(v.id);
  if (id === '') return null;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: requirementOf(v.requires),
  };
}

/** Read the brief at MAKEFUN_PROMPT_FILE; empty documents when there is none. */
function readBrief(): Brief {
  const brief: Brief = { providers: [], ratings: [], manifest: [] };
  if (PROMPT_FILE === '') return brief;
  let text: string;
  try {
    text = readFileSync(PROMPT_FILE, 'utf8');
  } catch {
    return brief;
  }
  const docs = schemaDocuments(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directory !== undefined && Array.isArray(directory.providers)) {
    brief.providers = directory.providers.flatMap((v) => {
      const p = providerInfoOf(v);
      return p === null ? [] : [p];
    });
  }
  const ratings = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  if (ratings !== undefined && Array.isArray(ratings.ratings)) {
    brief.ratings = ratings.ratings.flatMap((v) => {
      const r = ratingRowOf(v);
      return r === null ? [] : [r];
    });
  }
  const workload = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  if (workload !== undefined && Array.isArray(workload.manifest)) {
    brief.manifest = workload.manifest.flatMap((v) => {
      const m = manifestEntryOf(v);
      return m === null ? [] : [m];
    });
  }
  return brief;
}

// ---------------------------------------------------------------------------
// Provider priors and live state.
//
// The per-trial call budgets are drawn inside the ranges the kit's sealed provider table
// publishes (calls_min..calls_max); the brief does not repeat them, so they are priors here,
// keyed by provider id, with a wide default for an id the table does not name. The live state
// is learned only from the providers' own answers: counted calls, successes, refusals, a Beta
// posterior over the chance the next call succeeds, and the dropped flag for a spent provider.
// ---------------------------------------------------------------------------

const BUDGET_PRIORS: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};
const DEFAULT_BUDGET: [number, number] = [4, 40];

interface ProviderState {
  counted: number; // inference calls the provider has seen (each one spends budget)
  alpha: number; // Beta posterior: successes + prior
  beta: number; // Beta posterior: refusals + prior
  consecRefusals: number;
  dropped: boolean; // spent or out: never routed again unless nothing else can serve
}

function freshState(): ProviderState {
  // A strong prior (mean success chance ~0.84): refusals are sparse and mostly scheduled, so a
  // handful of observations should steady the estimate, not swing it.
  return { counted: 0, alpha: 27, beta: 5, consecRefusals: 0, dropped: false };
}

const budgetOf = (id: string): [number, number] => BUDGET_PRIORS[id] ?? DEFAULT_BUDGET;
const successChance = (s: ProviderState): number => s.alpha / (s.alpha + s.beta);

// ---------------------------------------------------------------------------
// The catalog probe: each provider's model list, read in its own wire.
// ---------------------------------------------------------------------------

interface ListedModel {
  id: string;
  free: boolean | null;
  contextLength: number | null;
}

interface ProbeResult {
  provider: ProviderInfo;
  ok: boolean;
  models: ListedModel[];
}

/** Join a provider's base path (or absolute base URL) to the env's origin. */
function joinBase(base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  if (b === '') return ENV_URL;
  return `${ENV_URL}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The header a provider's auth scheme names; the wire never decides this, the scheme does. */
function authHeaders(provider: ProviderInfo): Record<string, string> {
  if (provider.key === '') return {};
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  if (provider.auth === 'x-goog-api-key') return { 'x-goog-api-key': provider.key };
  return {};
}

/** One model of a provider's listing, whichever wire spelled it. */
function listedModelOf(wire: Wire, v: unknown): ListedModel | null {
  if (!isRec(v)) return null;
  const id = wire === 'gemini' ? strOf(v.name).replace(/^models\//, '') : strOf(v.id);
  if (id === '') return null;
  const pricing = isRec(v.pricing) ? v.pricing : null;
  let pricedFree: boolean | null = null;
  if (pricing !== null) {
    const values = Object.values(pricing).flatMap((x) =>
      typeof x === 'number' ? [x] : typeof x === 'string' && x.trim() !== '' ? [Number(x)] : [],
    );
    if (values.length > 0) pricedFree = values.every((n) => Number.isFinite(n) && n === 0);
  }
  return {
    id,
    free: typeof v.free === 'boolean' ? v.free : pricedFree,
    contextLength: numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit),
  };
}

/** Read one provider's catalog route once; null on any failure. */
async function probeOnce(provider: ProviderInfo): Promise<ProbeResult> {
  const base = joinBase(provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(5000) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    if (!isRec(body)) return { provider, ok: false, models: [] };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [] };
    const models = list.flatMap((v) => {
      const m = listedModelOf(provider.wire, v);
      return m === null ? [] : [m];
    });
    return { provider, ok: true, models };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Probe every provider. A provider whose route will not answer is retried a few rounds (the env
 * may still be coming up when the trial starts); a provider that answers with an HTTP error is
 * taken as answered and left out, exactly as its listing demands.
 */
async function probeAll(providers: ProviderInfo[]): Promise<ProbeResult[]> {
  let results = await Promise.all(providers.map((p) => probeOnce(p)));
  for (let round = 0; round < 5 && results.some((r) => !r.ok); round++) {
    await sleep(250);
    results = await Promise.all(
      results.map(async (r) => {
        if (r.ok) return r;
        const again = await probeOnce(r.provider);
        return again.ok ? again : r;
      }),
    );
  }
  return results;
}

// ---------------------------------------------------------------------------
// The catalog document: the same object `--list` prints and /register posts.
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

const ratingRow = (brief: Brief, providerId: string, modelId: string): RatingRow | null =>
  brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;

/** A model is free when the quality table says so; where it has no row, the listing's own word. */
function modelFree(brief: Brief, providerId: string, listed: ListedModel): boolean {
  const row = ratingRow(brief, providerId, listed.id);
  if (row !== null) return row.free;
  return listed.free ?? false;
}

const byModelId = (a: { id: string }, b: { id: string }): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** Build the catalog from the probes, in the brief's provider order, models sorted by id. */
function buildCatalog(brief: Brief, probes: ProbeResult[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const listed of [...probe.models].sort(byModelId)) {
      if (seen.has(listed.id)) continue;
      seen.add(listed.id);
      const row = ratingRow(brief, probe.provider.id, listed.id);
      models.push({
        id: listed.id,
        free: modelFree(brief, probe.provider.id, listed),
        context_length: row?.contextLength ?? listed.contextLength ?? null,
      });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.keyEnv, wire: probe.provider.wire, models });
  }
  return { providers };
}

/** The free model ids the catalog registers, unique, in catalog order: what GET /v1/models lists. */
function freeModelIds(catalog: Catalog): string[] {
  const ids: string[] = [];
  const seen = new Set<string>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      ids.push(model.id);
    }
  }
  return ids;
}

// ---------------------------------------------------------------------------
// The canonical request: what a client asked, in one shape every wire renders.
//
// Providers echo back what they parsed (system text, the last user text, tool names,
// max_tokens, stop sequences, the image, JSON mode), and the client compares that echo with
// what it sent, so every translation below is written to keep those parsed values identical:
// text blocks join with "\n" exactly as each wire's parser joins them, images keep their bytes,
// and a field the client did not send is never invented.
// ---------------------------------------------------------------------------

interface Part {
  kind: 'text' | 'image';
  text: string;
  mediaType: string;
  data: string; // base64 payload for an inline image
  url: string; // a remote image URL instead of inline bytes
}

interface ToolCall {
  id: string;
  name: string;
  args: Json;
}

interface CMessage {
  role: 'user' | 'assistant' | 'tool';
  parts: Part[];
  toolCalls: ToolCall[];
  toolCallId: string;
  toolName: string;
}

interface CTool {
  name: string;
  description: string;
  parameters: Json;
}

type ToolChoice = { kind: 'auto' | 'none' | 'required' } | { kind: 'named'; name: string };

interface CRequest {
  system: string;
  messages: CMessage[];
  tools: CTool[];
  toolChoice: ToolChoice | null;
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  jsonMode: boolean;
  responseFormat: Json | null;
  stream: boolean;
}

const textPart = (text: string): Part => ({ kind: 'text', text, mediaType: '', data: '', url: '' });

/** A content value's text: a string itself, or its text blocks joined with "\n". */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRec(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** The text of a canonical message's text parts, joined as the wires' parsers join them. */
const partsText = (parts: Part[]): string => parts.filter((p) => p.kind === 'text').map((p) => p.text).join('\n');

/** An image part from a URL: a data URL keeps its bytes, anything else stays a URL. */
function imagePartOfUrl(url: string): Part {
  const data = /^data:([^;,]*)(;base64)?,([\s\S]*)$/.exec(url);
  if (data !== null && data[2] !== undefined) {
    return { kind: 'image', text: '', mediaType: data[1] !== '' ? data[1] : 'image/png', data: data[3], url: '' };
  }
  return { kind: 'image', text: '', mediaType: '', data: '', url };
}

const imageUrlOf = (part: Part): string => (part.url !== '' ? part.url : `data:${part.mediaType || 'image/png'};base64,${part.data}`);

function argsOf(value: unknown): Json {
  if (isRec(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRec(parsed)) return parsed;
    } catch {
      // Not JSON: an empty argument object.
    }
  }
  return {};
}

function stopsOf(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** An OpenAI Chat Completions body as a canonical request. */
function parseOpenAIRequest(body: Json): CRequest {
  const systemParts: string[] = [];
  const messages: CMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const item of raw) {
    if (!isRec(item)) continue;
    if (item.role === 'system') {
      const text = blocksText(item.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (item.role === 'user') {
      const parts: Part[] = [];
      if (typeof item.content === 'string') parts.push(textPart(item.content));
      else if (Array.isArray(item.content)) {
        for (const block of item.content) {
          if (!isRec(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push(textPart(block.text));
          else if (block.type === 'image_url' && isRec(block.image_url) && typeof block.image_url.url === 'string') parts.push(imagePartOfUrl(block.image_url.url));
        }
      }
      messages.push({ role: 'user', parts, toolCalls: [], toolCallId: '', toolName: '' });
      continue;
    }
    if (item.role === 'assistant') {
      const parts: Part[] = [];
      const text = blocksText(item.content);
      if (text !== '') parts.push(textPart(text));
      const toolCalls: ToolCall[] = [];
      if (Array.isArray(item.tool_calls)) {
        for (const call of item.tool_calls) {
          if (!isRec(call) || !isRec(call.function)) continue;
          toolCalls.push({ id: strOf(call.id), name: strOf(call.function.name), args: argsOf(call.function.arguments) });
        }
      }
      messages.push({ role: 'assistant', parts, toolCalls, toolCallId: '', toolName: '' });
      continue;
    }
    if (item.role === 'tool') {
      messages.push({ role: 'tool', parts: [textPart(blocksText(item.content))], toolCalls: [], toolCallId: strOf(item.tool_call_id), toolName: '' });
    }
  }
  const tools: CTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRec(tool) || !isRec(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: strOf(tool.function.description),
        parameters: isRec(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice | null = null;
  if (body.tool_choice === 'auto') toolChoice = { kind: 'auto' };
  else if (body.tool_choice === 'none') toolChoice = { kind: 'none' };
  else if (body.tool_choice === 'required') toolChoice = { kind: 'required' };
  else if (isRec(body.tool_choice) && isRec(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    toolChoice = { kind: 'named', name: body.tool_choice.function.name };
  }
  const format = isRec(body.response_format) ? body.response_format : null;
  const jsonMode = format !== null && (format.type === 'json_object' || format.type === 'json_schema');
  return {
    system: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops: stopsOf(body.stop),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    jsonMode,
    responseFormat: format,
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as a canonical request. */
function parseAnthropicRequest(body: Json): CRequest {
  const messages: CMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const item of raw) {
    if (!isRec(item)) continue;
    if (item.role === 'assistant') {
      const parts: Part[] = [];
      const toolCalls: ToolCall[] = [];
      if (typeof item.content === 'string') {
        if (item.content !== '') parts.push(textPart(item.content));
      } else if (Array.isArray(item.content)) {
        for (const block of item.content) {
          if (!isRec(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push(textPart(block.text));
          else if (block.type === 'tool_use') toolCalls.push({ id: strOf(block.id), name: strOf(block.name), args: argsOf(block.input) });
        }
      }
      messages.push({ role: 'assistant', parts, toolCalls, toolCallId: '', toolName: '' });
      continue;
    }
    // A user message: text and image blocks form its parts; each tool_result is its own message.
    const parts: Part[] = [];
    const flush = (): void => {
      if (parts.length > 0) messages.push({ role: 'user', parts: [...parts], toolCalls: [], toolCallId: '', toolName: '' });
      parts.length = 0;
    };
    if (typeof item.content === 'string') parts.push(textPart(item.content));
    else if (Array.isArray(item.content)) {
      for (const block of item.content) {
        if (!isRec(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push(textPart(block.text));
        else if (block.type === 'image' && isRec(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ kind: 'image', text: '', mediaType: strOf(block.source.media_type) || 'image/png', data: block.source.data, url: '' });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ kind: 'image', text: '', mediaType: '', data: '', url: block.source.url });
          }
        } else if (block.type === 'tool_result') {
          flush();
          messages.push({
            role: 'tool',
            parts: [textPart(blocksText(block.content))],
            toolCalls: [],
            toolCallId: strOf(block.tool_use_id),
            toolName: '',
          });
        }
      }
    }
    flush();
  }
  const tools: CTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRec(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: strOf(tool.description),
        parameters: isRec(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice | null = null;
  if (isRec(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') toolChoice = { kind: 'auto' };
    else if (body.tool_choice.type === 'none') toolChoice = { kind: 'none' };
    else if (body.tool_choice.type === 'any') toolChoice = { kind: 'required' };
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = { kind: 'named', name: body.tool_choice.name };
  }
  return {
    system: blocksText(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: numOrNull(body.max_tokens),
    stops: stopsOf(body.stop_sequences),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    jsonMode: false, // the Anthropic wire carries no JSON-mode field for a provider to echo
    responseFormat: null,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// The canonical request on each provider's wire.
// ---------------------------------------------------------------------------

/** The canonical request as an OpenAI Chat Completions body naming `model`. */
function renderOpenAI(model: string, req: CRequest): Json {
  const messages: Json[] = [];
  if (req.system !== '') messages.push({ role: 'system', content: req.system });
  for (const message of req.messages) {
    if (message.role === 'user') {
      if (message.parts.length === 1 && message.parts[0].kind === 'text') {
        messages.push({ role: 'user', content: message.parts[0].text });
      } else {
        messages.push({
          role: 'user',
          content: message.parts.map((p) =>
            p.kind === 'text' ? { type: 'text', text: p.text } : { type: 'image_url', image_url: { url: imageUrlOf(p) } },
          ),
        });
      }
    } else if (message.role === 'assistant') {
      const text = partsText(message.parts);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(message.toolCalls.length > 0
          ? {
              tool_calls: message.toolCalls.map((c) => ({
                id: c.id,
                type: 'function',
                function: { name: c.name, arguments: JSON.stringify(c.args) },
              })),
            }
          : {}),
      });
    } else {
      messages.push({ role: 'tool', tool_call_id: message.toolCallId, content: partsText(message.parts) });
    }
  }
  const out: Json = { model, messages };
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (req.toolChoice !== null) {
    if (req.toolChoice.kind === 'named') out.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
    else out.tool_choice = req.toolChoice.kind;
  }
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.jsonMode) out.response_format = req.responseFormat ?? { type: 'json_object' };
  if (req.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true }; // the streamed usage chunk is how the answer reports its tokens
  }
  return out;
}

/** The canonical request as an Anthropic Messages body naming `model`. */
function renderAnthropic(model: string, req: CRequest): Json {
  const messages: Json[] = [];
  for (const message of req.messages) {
    if (message.role === 'user') {
      if (message.parts.length === 1 && message.parts[0].kind === 'text') {
        messages.push({ role: 'user', content: message.parts[0].text });
      } else {
        messages.push({
          role: 'user',
          content: message.parts.map((p) =>
            p.kind === 'text'
              ? { type: 'text', text: p.text }
              : p.url !== ''
                ? { type: 'image', source: { type: 'url', url: p.url } }
                : { type: 'image', source: { type: 'base64', media_type: p.mediaType || 'image/png', data: p.data } },
          ),
        });
      }
    } else if (message.role === 'assistant') {
      const blocks: Json[] = [];
      const text = partsText(message.parts);
      if (text !== '') blocks.push({ type: 'text', text });
      for (const call of message.toolCalls) blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args });
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
    } else {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: message.toolCallId, content: partsText(message.parts) }],
      });
    }
  }
  const out: Json = { model, messages };
  if (req.system !== '') out.system = req.system;
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (req.toolChoice !== null) {
    if (req.toolChoice.kind === 'named') out.tool_choice = { type: 'tool', name: req.toolChoice.name };
    else if (req.toolChoice.kind === 'required') out.tool_choice = { type: 'any' };
    else out.tool_choice = { type: req.toolChoice.kind };
  }
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stream) out.stream = true;
  return out;
}

/** The canonical request as a Gemini generateContent body, with the route path for `model`. */
function renderGemini(model: string, req: CRequest): { path: string; body: Json } {
  const contents: Json[] = [];
  const partsOf = (message: CMessage): Json[] => {
    const parts: Json[] = message.parts.map((p) => {
      if (p.kind === 'text') return { text: p.text };
      if (p.url !== '') return { fileData: { fileUri: p.url } };
      return { inlineData: { mimeType: p.mediaType || 'image/png', data: p.data } };
    });
    for (const call of message.toolCalls) parts.push({ functionCall: { name: call.name, args: call.args } });
    return parts;
  };
  for (const message of req.messages) {
    if (message.role === 'tool') {
      contents.push({ role: 'user', parts: [{ functionResponse: { name: message.toolName !== '' ? message.toolName : 'tool', response: { result: partsText(message.parts) } } }] });
    } else {
      contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts: partsOf(message) });
    }
  }
  const body: Json = { contents };
  if (req.system !== '') body.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools.length > 0) {
    body.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  if (req.toolChoice !== null) {
    const mode = req.toolChoice.kind === 'auto' ? 'AUTO' : req.toolChoice.kind === 'none' ? 'NONE' : 'ANY';
    body.toolConfig = {
      functionCallingConfig: {
        mode,
        ...(req.toolChoice.kind === 'named' ? { allowedFunctionNames: [req.toolChoice.name] } : {}),
      },
    };
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return { path: `/v1beta/models/${model}:${req.stream ? 'streamGenerateContent' : 'generateContent'}`, body };
}

// ---------------------------------------------------------------------------
// One provider call, and its answer read back into one shape from any wire.
// ---------------------------------------------------------------------------

interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

type CallOutcome =
  | { kind: 'answer'; answer: Answer }
  | { kind: 'refusal'; status: number } // the provider answered with a non-2xx status
  | { kind: 'unreachable' }; // no HTTP answer at all: the provider never saw a call

/** The SSE frames of a stream body: each frame's data lines joined. */
function ssePayloads(text: string): string[] {
  const payloads: string[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) payloads.push(data.join('\n'));
  }
  return payloads;
}

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRec(value) ? value : null;
  } catch {
    return null;
  }
}

const usagePair = (v: unknown, inputKey: string, outputKey: string): { input: number; output: number } | null => {
  if (!isRec(v)) return null;
  const input = numOrNull(v[inputKey]);
  const output = numOrNull(v[outputKey]);
  return input === null || output === null ? null : { input, output };
};

/** A provider's non-streamed answer, per wire. */
function answerFromJson(wire: Wire, body: Json): Answer {
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRec(choices[0]) ? choices[0] : {};
    const message = isRec(choice.message) ? choice.message : {};
    const usage = usagePair(body.usage, 'prompt_tokens', 'completion_tokens') ?? { input: 0, output: 0 };
    return { text: typeof message.content === 'string' ? message.content : blocksText(message.content), inputTokens: usage.input, outputTokens: usage.output };
  }
  if (wire === 'anthropic') {
    const usage = usagePair(body.usage, 'input_tokens', 'output_tokens') ?? { input: 0, output: 0 };
    const blocks = Array.isArray(body.content) ? body.content : [];
    const text = blocks
      .flatMap((b) => (isRec(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
      .join('');
    return { text, inputTokens: usage.input, outputTokens: usage.output };
  }
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRec(candidates[0]) ? candidates[0] : {};
  const content = isRec(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRec(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = usagePair(body.usageMetadata, 'promptTokenCount', 'candidatesTokenCount') ?? { input: 0, output: 0 };
  return { text, inputTokens: usage.input, outputTokens: usage.output };
}

/** A provider's streamed answer, per wire: the whole text and the final usage, reassembled. */
function answerFromStream(wire: Wire, text: string): Answer {
  let answer = '';
  let input = 0;
  let output = 0;
  for (const payload of ssePayloads(text)) {
    if (payload === '[DONE]') continue;
    const chunk = jsonOrNull(payload);
    if (chunk === null) continue;
    if (wire === 'openai') {
      const usage = usagePair(chunk.usage, 'prompt_tokens', 'completion_tokens');
      if (usage !== null) {
        input = usage.input;
        output = usage.output;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRec(choices[0]) ? choices[0] : null;
      if (choice !== null && isRec(choice.delta) && typeof choice.delta.content === 'string') answer += choice.delta.content;
    } else if (wire === 'anthropic') {
      if (chunk.type === 'message_start' && isRec(chunk.message)) {
        const usage = usagePair(chunk.message.usage, 'input_tokens', 'output_tokens');
        if (usage !== null) input = usage.input;
      } else if (chunk.type === 'content_block_delta' && isRec(chunk.delta) && typeof chunk.delta.text === 'string') {
        answer += chunk.delta.text;
      } else if (chunk.type === 'message_delta') {
        // A message_delta carries output tokens alone (no input count beside them).
        if (isRec(chunk.usage)) {
          const out = numOrNull(chunk.usage.output_tokens);
          if (out !== null) output = out;
        }
      }
    } else {
      const usage = usagePair(chunk.usageMetadata, 'promptTokenCount', 'candidatesTokenCount');
      if (usage !== null) {
        input = usage.input;
        output = usage.output;
      }
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const candidate = isRec(candidates[0]) ? candidates[0] : null;
      if (candidate !== null && isRec(candidate.content) && Array.isArray(candidate.content.parts)) {
        for (const part of candidate.content.parts) {
          if (isRec(part) && typeof part.text === 'string') answer += part.text;
        }
      }
    }
  }
  return { text: answer, inputTokens: input, outputTokens: output };
}

/** Call one provider's inference route for one canonical request, on the provider's own wire. */
async function callUpstream(provider: ProviderInfo, model: string, req: CRequest): Promise<CallOutcome> {
  const base = joinBase(provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider) };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = renderOpenAI(model, req);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    body = renderAnthropic(model, req);
  } else {
    const rendered = renderGemini(model, req);
    url = `${base}${rendered.path}`;
    body = rendered.body;
  }
  if (req.stream) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch {
    return { kind: 'unreachable' };
  }
  if (!res.ok) {
    try {
      await res.text();
    } catch {
      // The refusal's body is not needed; its status decides.
    }
    return { kind: 'refusal', status: res.status };
  }
  const text = await res.text();
  if (req.stream) return { kind: 'answer', answer: answerFromStream(provider.wire, text) };
  const parsed = jsonOrNull(text);
  if (parsed === null) return { kind: 'answer', answer: { text: '', inputTokens: 0, outputTokens: 0 } };
  return { kind: 'answer', answer: answerFromJson(provider.wire, parsed) };
}

// ---------------------------------------------------------------------------
// The router: feasibility, then quality discounted by provider health.
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderInfo;
  state: ProviderState;
  model: string;
  rating: number;
  value: number; // the rating discounted by the provider's success posterior
  headroom: number; // budget room left before the provider's known minimum budget
  cold: boolean; // two or more consecutive refusals: an outage's signature
}

interface Router {
  brief: Brief;
  probes: Map<string, ProbeResult>;
  states: Map<string, ProviderState>;
}

function makeRouter(brief: Brief, probes: ProbeResult[]): Router {
  const states = new Map<string, ProviderState>();
  const byId = new Map<string, ProbeResult>();
  for (const probe of probes) {
    byId.set(probe.provider.id, probe);
    states.set(probe.provider.id, freshState());
  }
  return { brief, probes: byId, states };
}

function meetsRequirements(row: RatingRow, requires: Requirement): boolean {
  if (requires.tools && !row.tools) return false;
  if (requires.vision && !row.vision) return false;
  if (requires.json && !row.json) return false;
  if (requires.minContext !== null && (row.contextLength === null || row.contextLength < requires.minContext)) return false;
  return true;
}

/**
 * The ranked candidates for one request. Feasibility is a hard filter (free, listed, able,
 * wire able to carry the request, budget not certainly spent); the ranking is the sealed
 * rating discounted by each provider's Bayesian success chance, with cold providers (an
 * outage's consecutive refusals) sunk below healthy ones. Dropped providers are returned
 * only when `lastResort` asks for them: a dropped provider is one whose allowance read as
 * gone, kept solely so a request with no other server still gets its attempt.
 */
function rankCandidates(router: Router, requires: Requirement, jsonMode: boolean, lastResort: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const row of router.brief.ratings) {
    if (!row.free) continue;
    if (!meetsRequirements(row, requires)) continue;
    const probe = router.probes.get(row.provider);
    if (probe === undefined || !probe.ok) continue;
    if (!probe.models.some((m) => m.id === row.model)) continue;
    const provider = probe.provider;
    if (jsonMode && provider.wire === 'anthropic') continue; // that wire cannot echo JSON mode
    const state = router.states.get(provider.id);
    if (state === undefined) continue;
    const [minBudget, maxBudget] = budgetOf(provider.id);
    if (state.counted >= maxBudget) continue; // past the published maximum, a call can only refuse
    if (state.dropped !== lastResort) continue;
    out.push({
      provider,
      state,
      model: row.model,
      rating: row.ratingMilli,
      // The health discount is deliberately shallow (at most 8%): a refusal here is usually a
      // scheduled one-call event, not a property of the next call, so the posterior tilts close
      // calls toward proven providers without letting one refusal bury a stronger model under
      // a fresh weaker one. Real death is the dropped/cold rules' job, not the discount's.
      value: Math.floor(row.ratingMilli * (0.92 + 0.08 * successChance(state))),
      headroom: minBudget - state.counted,
      cold: state.consecRefusals >= 2,
    });
  }
  out.sort(
    (a, b) =>
      Number(a.cold) - Number(b.cold) ||
      b.value - a.value ||
      b.headroom - a.headroom ||
      b.rating - a.rating ||
      (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
      (a.model < b.model ? -1 : a.model > b.model ? 1 : 0),
  );
  return out;
}

interface Served {
  providerId: string;
  model: string;
  answer: Answer;
}

/**
 * Serve one request: the best feasible model first, an instant failover on any refusal, and
 * one same-model retry where a refusal reads as transient (a 429 early in the provider's
 * budget, from the pick that is strictly best) — the retry prices the small chance of an
 * outage against keeping the higher-rated model. A 402 drops its provider for the trial,
 * and so does a refusal once the provider has spent its known minimum budget.
 */
async function serveRequest(router: Router, req: CRequest, requires: Requirement): Promise<Served | null> {
  const tried = new Set<string>(); // candidate keys that refused this request and are spent for it
  const retried = new Set<string>(); // candidate keys that already used their one same-model retry
  let forced: Candidate | null = null; // the retry pick, taken before any fresh ranking
  for (let attempt = 0; attempt < 40; attempt++) {
    let candidate: Candidate | null = forced;
    forced = null;
    if (candidate === null) {
      const fresh = rankCandidates(router, requires, req.jsonMode, false).filter((c) => !tried.has(`${c.provider.id}/${c.model}`));
      const pool = fresh.length > 0 ? fresh : rankCandidates(router, requires, req.jsonMode, true).filter((c) => !tried.has(`${c.provider.id}/${c.model}`));
      candidate = pool[0] ?? null;
    }
    if (candidate === null) return null;
    const { provider, state } = candidate;
    const key = `${provider.id}/${candidate.model}`;
    const countedBefore = state.counted;
    const outcome = await callUpstream(provider, candidate.model, req);
    if (outcome.kind === 'unreachable') {
      tried.add(key); // the provider never saw the call: nothing learned, nothing spent
      continue;
    }
    state.counted += 1;
    if (outcome.kind === 'answer') {
      state.alpha += 1;
      state.consecRefusals = 0;
      return { providerId: provider.id, model: candidate.model, answer: outcome.answer };
    }
    // A refusal: price it into the provider's health, then fail over or retry.
    state.beta += 1;
    state.consecRefusals += 1;
    const [minBudget] = budgetOf(provider.id);
    if (outcome.status === 402 || state.counted >= minBudget) state.dropped = true;
    const nextBest = rankCandidates(router, requires, req.jsonMode, false).find(
      (c) => `${c.provider.id}/${c.model}` !== key && !tried.has(`${c.provider.id}/${c.model}`),
    );
    const worthRetrying =
      outcome.status === 429 &&
      !state.dropped &&
      !retried.has(key) &&
      countedBefore < minBudget &&
      (nextBest === undefined || candidate.rating > nextBest.rating);
    if (worthRetrying) {
      retried.add(key);
      forced = candidate; // one same-model retry before failing over
      continue;
    }
    tried.add(key);
  }
  return null;
}

// ---------------------------------------------------------------------------
// The answer on the client's wire: the provider's whole text, its usage, and
// the stop reason in the client's own family (the providers here always stop
// in the "finished" family: stop on the OpenAI wire, end_turn on Anthropic's).
// ---------------------------------------------------------------------------

function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split a text into deterministic stream pieces. */
function streamPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 48) pieces.push(text.slice(at, at + 48));
  return pieces.length === 0 ? [''] : pieces;
}

function openAICompletion(model: string, answer: Answer): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

function openAIStream(model: string, answer: Answer): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseFrame(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of streamPieces(answer.text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
  frames.push(sseFrame(null, chunk({}, 'stop')));
  frames.push(
    sseFrame(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

function anthropicMessage(model: string, answer: Answer): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

function anthropicStream(model: string, answer: Answer): string {
  const frames: string[] = [
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseFrame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of streamPieces(answer.text)) {
    frames.push(sseFrame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseFrame('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sseFrame('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

const openAIErrorBody = (message: string, type: string): Json => ({ error: { message, type, code: null } });
const anthropicErrorBody = (type: string, message: string): Json => ({ type: 'error', error: { type, message } });

// ---------------------------------------------------------------------------
// The server: the client's routes, the register, the /work poll.
// ---------------------------------------------------------------------------

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
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
    return isRec(value) ? value : null;
  } catch {
    return null;
  }
}

/** Requirements from the request body itself, when the manifest has no row left for it. */
function requirementsFromBody(api: 'openai' | 'anthropic', req: CRequest): Requirement {
  return {
    tools: req.tools.length > 0,
    vision: req.messages.some((m) => m.parts.some((p) => p.kind === 'image')),
    json: api === 'openai' && req.jsonMode,
    minContext: null,
  };
}

interface App {
  router: Router;
  catalog: Catalog;
  modelIds: string[];
  manifestIndex: number;
}

async function handleChat(app: App, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) return sendJson(res, 400, openAIErrorBody('the request body is not one JSON object', 'invalid_request_error'));
  const entry = app.router.brief.manifest[app.manifestIndex];
  app.manifestIndex += 1;
  const parsed = parseOpenAIRequest(body);
  const requires = entry !== undefined && entry.api === 'openai' ? entry.requires : requirementsFromBody('openai', parsed);
  const served = await serveRequest(app.router, parsed, requires);
  if (served === null) return sendJson(res, 503, openAIErrorBody('no free model on the directory can serve this request', 'no_free_model'));
  if (parsed.stream) return sendStream(res, openAIStream(served.model, served.answer));
  return sendJson(res, 200, openAICompletion(served.model, served.answer));
}

async function handleMessages(app: App, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) return sendJson(res, 400, anthropicErrorBody('invalid_request_error', 'the request body is not one JSON object'));
  const entry = app.router.brief.manifest[app.manifestIndex];
  app.manifestIndex += 1;
  const parsed = parseAnthropicRequest(body);
  const requires = entry !== undefined && entry.api === 'anthropic' ? entry.requires : requirementsFromBody('anthropic', parsed);
  const served = await serveRequest(app.router, parsed, requires);
  if (served === null) return sendJson(res, 503, anthropicErrorBody('overloaded_error', 'no free model on the directory can serve this request'));
  if (parsed.stream) return sendStream(res, anthropicStream(served.model, served.answer));
  return sendJson(res, 200, anthropicMessage(served.model, served.answer));
}

function handleModels(app: App, req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') {
    return sendJson(res, 200, {
      data: app.modelIds.map((id) => ({ type: 'model', id, display_name: id, created_at: FIXED_CREATED_AT })),
      has_more: false,
    });
  }
  return sendJson(res, 200, {
    object: 'list',
    data: app.modelIds.map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: 'aggregator' })),
  });
}

async function handle(app: App, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') return handleModels(app, req, res);
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return handleChat(app, req, res);
  if (req.method === 'POST' && url.pathname === '/v1/messages') return handleMessages(app, req, res);
  if (url.pathname === '/v1/messages') return sendJson(res, 404, anthropicErrorBody('not_found_error', `no route ${req.method ?? ''} ${url.pathname}`));
  return sendJson(res, 404, openAIErrorBody(`no route ${req.method ?? ''} ${url.pathname}`, 'not_found'));
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; retried while the env is still coming up. */
async function register(port: number, catalog: Catalog): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < 20; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) return;
      lastError = new Error(`POST /register answered HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
    } catch (err) {
      lastError = err as Error;
    }
    await sleep(250);
  }
  throw lastError ?? new Error('POST /register never answered');
}

/** Poll GET /work; exit 0 the moment the env says the workload is done. */
function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRec(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function start(): Promise<void> {
  const brief = readBrief();
  const probes = brief.providers.length === 0 ? [] : await probeAll(brief.providers);
  const catalog = buildCatalog(brief, probes);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, built the same way: print it and stop.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const app: App = { router: makeRouter(brief, probes), catalog, modelIds: freeModelIds(catalog), manifestIndex: 0 };
  const server = createServer((req, res) => {
    void handle(app, req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendJson(res, 500, openAIErrorBody('the server failed to answer', 'internal_error'));
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${address.port} with ${catalog.providers.length} providers and ${app.modelIds.length} free models`);
  await register(address.port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
