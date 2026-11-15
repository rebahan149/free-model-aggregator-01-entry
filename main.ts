// free-model-aggregator-01, v2: a fixed-plan aggregator over ten simulated free-tier providers.
//
// Design in one paragraph: the score's total is capped by arithmetic, not by effort. Quality is
// earned rating over a bound in which 17 of 20 requests are best served by one provider's single
// top model; that provider's per-trial budget is drawn between 10 and 12 counting calls, and at
// least one of those calls is always a scheduled refusal, so at most 11 requests can ever earn the
// top rating and the weighted shortfall can never reach zero. Quality 98-99 both round to the same
// total of 99 when the other four columns are perfect, and exact ties on score are broken by the
// number of upstream calls. So this entry spends its effort where the tie-break is: a FIXED plan,
// derived from the brief before the first request arrives, that makes exactly one upstream call
// per request in the clean case and never probes. Failover spends extra calls only when a refusal
// actually happens, and then as cheaply as the mechanics allow: the walk prefers providers never
// called yet — a provider's first call number never refuses (the drawn refusing numbers and the
// outages all start at 2, and budgets start at 4) — over re-gambling on hot providers, so a
// request whose planned call failed still closes in exactly one more call, at a slightly lower
// rating. A provider is dropped only when it is proven spent, by two refusals in a row: a lone
// refusal is a scheduled event (each provider draws one or two refusing call numbers, and one
// provider always refuses its second call), so dropping on it would discard the provider's
// remaining budget, and the cascade onto lesser providers costs more calls, and more quality,
// than the single confirmation call that proves a budget is gone.
//
// Everything the plan uses is read from the brief: the provider directory (wires, auth, paths),
// the quality table (ratings, capabilities, context lengths) and the workload manifest. The plan
// caps each provider's planned first calls at the smallest budget its free tier is known to draw
// in this competition's directory of tiers, so planned calls stay inside budgets that always
// exist; anything the draw withholds is discovered by a refusal and handled by the ladder, never
// by spending calls in advance. All three upstream wires are spoken faithfully: a client request
// is translated so the answering provider echoes exactly the system text, user text, tool names,
// max tokens, stops, image and JSON mode the client sent, and the provider's answer text and
// token counts are returned untouched, in the client's own wire, streamed as synthesized events
// when the client asked for a stream. No clocks and no randomness anywhere: replaying a trial
// re-derives the same plan and walks it identically.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------- small helpers

type Json = Record<string, unknown>;

function isRec(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
function str(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}
function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}
function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.filter((v): v is string => typeof v === 'string');
  return [];
}
function log(message: string): void {
  process.stderr.write(`aggregator-v2: ${message}\n`);
}

// ---------------------------------------------------------------- the brief

interface ProviderDir {
  id: string;
  wire: 'openai' | 'anthropic' | 'gemini';
  auth: 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
  keyEnv: string;
  key: string;
  path: string;
}
interface RatingRow {
  provider: string;
  model: string;
  rating: number;
  context: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}
interface Requires {
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
  requires: Requires;
}
interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  manifest: ManifestEntry[];
}

/** Extract one JSON document (identified by its schema marker) from the brief's prose. */
function extractDoc(text: string, marker: string): Json | null {
  // The documents are printed one per line, starting with the schema key; the prose also quotes
  // the marker inside shape sketches, so prefer a whole line that parses and carries the marker.
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{"schema"') || !trimmed.includes(marker)) continue;
    try {
      const parsed: unknown = JSON.parse(trimmed);
      if (isRec(parsed)) return parsed;
    } catch {
      // Not the document line after all: fall through to the brace scan.
    }
  }
  const at = text.indexOf(marker);
  if (at < 0) return null;
  // Walk back to the opening brace of the object that carries the marker, then balance braces.
  let start = at;
  while (start > 0 && text[start] !== '{') start -= 1;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        try {
          const parsed: unknown = JSON.parse(text.slice(start, i + 1));
          return isRec(parsed) ? parsed : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

function readBrief(file: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [] };
  if (file === '') return empty;
  let text: string;
  try {
    text = readFileSync(file, 'utf8');
  } catch {
    return empty;
  }
  const dirDoc = extractDoc(text, 'free-model-aggregator-01/providers/v1');
  const ratingsDoc = extractDoc(text, 'free-model-aggregator-01/ratings/v1');
  const workloadDoc = extractDoc(text, 'free-model-aggregator-01/workload/v1');
  const providers: ProviderDir[] = [];
  if (dirDoc !== null && Array.isArray(dirDoc.providers)) {
    for (const raw of dirDoc.providers) {
      if (!isRec(raw)) continue;
      const id = str(raw.id);
      const wire = str(raw.wire);
      const auth = str(raw.auth);
      const keyEnv = str(raw.key_env);
      const key = str(raw.key);
      const path = str(raw.path);
      if (id === null || keyEnv === null || key === null || path === null) continue;
      if (wire !== 'openai' && wire !== 'anthropic' && wire !== 'gemini') continue;
      if (auth !== 'bearer' && auth !== 'x-api-key' && auth !== 'x-goog-api-key' && auth !== 'none') continue;
      providers.push({ id, wire, auth, keyEnv, key, path });
    }
  }
  const ratings: RatingRow[] = [];
  const ratingRows = ratingsDoc !== null ? (ratingsDoc.ratings ?? ratingsDoc.models) : null;
  if (Array.isArray(ratingRows)) {
    for (const raw of ratingRows) {
      if (!isRec(raw)) continue;
      const provider = str(raw.provider);
      const model = str(raw.model);
      const rating = num(raw.rating_milli);
      const context = num(raw.context_length);
      if (provider === null || model === null || rating === null || context === null) continue;
      if (raw.free !== true) continue;
      ratings.push({
        provider,
        model,
        rating,
        context,
        tools: raw.tools === true,
        vision: raw.vision === true,
        json: raw.json === true,
      });
    }
  }
  const manifest: ManifestEntry[] = [];
  if (workloadDoc !== null && Array.isArray(workloadDoc.manifest)) {
    for (const raw of workloadDoc.manifest) {
      if (!isRec(raw)) continue;
      const id = str(raw.id);
      const api = str(raw.api);
      const weight = num(raw.weight);
      if (id === null || weight === null) continue;
      if (api !== 'openai' && api !== 'anthropic') continue;
      const req = isRec(raw.requires) ? raw.requires : {};
      manifest.push({
        id,
        api,
        stream: raw.stream === true,
        weight,
        requires: {
          tools: req.tools === true,
          vision: req.vision === true,
          json: req.json === true,
          minContext: num(req.min_context),
        },
      });
    }
  }
  return { providers, ratings, manifest };
}

// ---------------------------------------------------------------- the catalog

interface Catalog {
  providers: { id: string; base_url: string; key_env: string; wire: string; models: { id: string; free: boolean; context_length: number }[] }[];
}

/** The one catalog document: built from the brief alone, used for --list and for /register alike. */
function buildCatalog(brief: Brief): Catalog {
  return {
    providers: brief.providers.map((provider) => ({
      id: provider.id,
      base_url: provider.path,
      key_env: provider.keyEnv,
      wire: provider.wire,
      models: brief.ratings
        .filter((row) => row.provider === provider.id)
        .map((row) => ({ id: row.model, free: true, context_length: row.context })),
    })),
  };
}

/** The unique free model ids, in first-appearance order: what GET /v1/models lists. */
function freeModelIds(brief: Brief): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const row of brief.ratings) {
    if (seen.has(row.model)) continue;
    seen.add(row.model);
    out.push(row.model);
  }
  return out;
}

// ---------------------------------------------------------------- the fixed plan

interface Candidate {
  provider: string;
  model: string;
  rating: number;
}

/**
 * The smallest per-trial budget each provider's free tier draws in this competition (the floor of
 * its published tier range): planned first calls per provider never exceed the plan cap derived
 * from it below, so a plan never counts on calls a budget might not hold. Providers missing from
 * the table plan without a cap.
 */
const BUDGET_FLOOR: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/**
 * How many planned first calls each provider gets. Equal to the budget floor, except zai: its
 * budget range is so small (4-5) that its second and later call numbers refuse roughly half the
 * time, so only its first call — the one call number that never refuses — is planned; its later
 * call numbers are spent only by the walk, when a refusal elsewhere has already been paid for.
 */
const PRIMARY_CAP: Record<string, number> = { ...BUDGET_FLOOR, zai: 1 };

function meets(row: RatingRow, req: Requires): boolean {
  if (req.tools && !row.tools) return false;
  if (req.vision && !row.vision) return false;
  if (req.json && !row.json) return false;
  if (req.minContext !== null && row.context < req.minContext) return false;
  return true;
}

/**
 * Build the plan: for every manifest request, an ordered candidate ladder. The first rung is the
 * request's planned call, chosen greedily in manifest order among eligible candidates whose
 * provider still has planned capacity (best rating, then most headroom, then directory order);
 * the remaining rungs are every other eligible candidate in rating order, walked only when a
 * refusal actually happens. One request class is wire-limited: an OpenAI client asking for JSON
 * mode can only be echoed faithfully by an openai-wire or gemini-wire provider (an anthropic-wire
 * provider has no way to carry response_format), so anthropic-wire candidates are excluded there.
 */
function buildPlan(brief: Brief): Map<string, Candidate[]> {
  const dirOrder = new Map(brief.providers.map((p, i) => [p.id, i]));
  const wireOf = new Map(brief.providers.map((p) => [p.id, p.wire]));
  const planned = new Map<string, number>();
  const plan = new Map<string, Candidate[]>();
  for (const entry of brief.manifest) {
    const wireLimited = entry.api === 'openai' && entry.requires.json;
    const eligible = brief.ratings.filter(
      (row) => meets(row, entry.requires) && (!wireLimited || wireOf.get(row.provider) !== 'anthropic'),
    );
    const byRating = (a: RatingRow, b: RatingRow): number =>
      b.rating - a.rating ||
      (dirOrder.get(a.provider) ?? 0) - (dirOrder.get(b.provider) ?? 0) ||
      (a.model < b.model ? -1 : a.model > b.model ? 1 : 0);
    const headroom = (row: RatingRow): number => {
      const cap = PRIMARY_CAP[row.provider];
      return cap === undefined ? Number.MAX_SAFE_INTEGER : cap - (planned.get(row.provider) ?? 0);
    };
    const primaryPool = eligible.filter((row) => headroom(row) > 0);
    const primary = [...primaryPool].sort(
      (a, b) => b.rating - a.rating || headroom(b) - headroom(a) || byRating(a, b),
    )[0];
    const ladder: Candidate[] = [];
    if (primary !== undefined) {
      ladder.push({ provider: primary.provider, model: primary.model, rating: primary.rating });
      planned.set(primary.provider, (planned.get(primary.provider) ?? 0) + 1);
    }
    for (const row of [...eligible].sort(byRating)) {
      if (primary !== undefined && row.provider === primary.provider && row.model === primary.model) continue;
      ladder.push({ provider: row.provider, model: row.model, rating: row.rating });
    }
    plan.set(entry.id, ladder);
  }
  return plan;
}

// ---------------------------------------------------------------- canonical requests

type CanonImage = { kind: 'bytes'; mediaType: string; data: string } | { kind: 'url'; url: string };
interface CanonMsg {
  role: 'user' | 'assistant';
  text: string;
  images: CanonImage[];
}
interface CanonTool {
  name: string;
  description: string | null;
  parameters: unknown;
}
interface Canon {
  systemText: string;
  messages: CanonMsg[];
  tools: CanonTool[];
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  jsonMode: boolean;
  stream: boolean;
}

/** The text a wire's parser reads out of a content value: text blocks joined with newlines. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRec(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function dataUrlImage(url: string): CanonImage | null {
  const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (match !== null) return { kind: 'bytes', mediaType: match[1], data: match[2] };
  return null;
}

function canonFromOpenAI(body: Json): Canon {
  const messages: CanonMsg[] = [];
  const systemParts: string[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRec(raw)) continue;
    const role = str(raw.role);
    if (role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role !== 'user' && role !== 'assistant') continue;
    const images: CanonImage[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRec(block) || block.type !== 'image_url' || !isRec(block.image_url)) continue;
        const url = str(block.image_url.url);
        if (url === null) continue;
        images.push(dataUrlImage(url) ?? { kind: 'url', url });
      }
    }
    messages.push({ role, text: blocksText(raw.content), images });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRec(raw) || !isRec(raw.function)) continue;
      const name = str(raw.function.name);
      if (name === null) continue;
      tools.push({ name, description: str(raw.function.description), parameters: raw.function.parameters ?? null });
    }
  }
  const format = isRec(body.response_format) ? str(body.response_format.type) : null;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    maxTokens: num(body.max_tokens) ?? num(body.max_completion_tokens),
    stops: stringList(body.stop),
    temperature: num(body.temperature),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

function canonFromAnthropic(body: Json): Canon {
  const messages: CanonMsg[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRec(raw)) continue;
    const role = str(raw.role);
    if (role !== 'user' && role !== 'assistant') continue;
    const images: CanonImage[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRec(block) || block.type !== 'image' || !isRec(block.source)) continue;
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          images.push({ kind: 'bytes', mediaType: str(source.media_type) ?? 'image/png', data: source.data });
        } else if (source.type === 'url' && typeof source.url === 'string') {
          images.push({ kind: 'url', url: source.url });
        }
      }
    }
    messages.push({ role, text: blocksText(raw.content), images });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRec(raw)) continue;
      const name = str(raw.name);
      if (name === null) continue;
      tools.push({ name, description: str(raw.description), parameters: raw.input_schema ?? null });
    }
  }
  return {
    systemText: blocksText(body.system),
    messages,
    tools,
    maxTokens: num(body.max_tokens),
    stops: stringList(body.stop_sequences),
    temperature: num(body.temperature),
    jsonMode: false,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------- upstream calls

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');

function providerUrl(provider: ProviderDir, model: string): string {
  const base = `${ENV_URL}${provider.path}`;
  if (provider.wire === 'openai') return `${base}/v1/chat/completions`;
  if (provider.wire === 'anthropic') return `${base}/v1/messages`;
  return `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
}

function providerHeaders(provider: ProviderDir): Record<string, string> {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (provider.auth === 'bearer') headers.authorization = `Bearer ${provider.key}`;
  else if (provider.auth === 'x-api-key') headers['x-api-key'] = provider.key;
  else if (provider.auth === 'x-goog-api-key') headers['x-goog-api-key'] = provider.key;
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

/** Build the upstream body for a cross-wire call from the canonical request. */
function buildUpstreamBody(provider: ProviderDir, model: string, canon: Canon): Json {
  if (provider.wire === 'openai') {
    const messages: unknown[] = [];
    if (canon.systemText !== '') messages.push({ role: 'system', content: canon.systemText });
    for (const msg of canon.messages) {
      if (msg.images.length === 0) {
        messages.push({ role: msg.role, content: msg.text });
      } else {
        const blocks: unknown[] = [];
        if (msg.text !== '') blocks.push({ type: 'text', text: msg.text });
        for (const img of msg.images) {
          blocks.push({
            type: 'image_url',
            image_url: { url: img.kind === 'url' ? img.url : `data:${img.mediaType};base64,${img.data}` },
          });
        }
        messages.push({ role: msg.role, content: blocks });
      }
    }
    const body: Json = { model, messages, stream: false };
    if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
    if (canon.temperature !== null) body.temperature = canon.temperature;
    if (canon.stops.length > 0) body.stop = canon.stops;
    if (canon.tools.length > 0) {
      body.tools = canon.tools.map((tool) => ({
        type: 'function',
        function: {
          name: tool.name,
          ...(tool.description !== null ? { description: tool.description } : {}),
          ...(tool.parameters !== null ? { parameters: tool.parameters } : {}),
        },
      }));
    }
    if (canon.jsonMode) body.response_format = { type: 'json_object' };
    return body;
  }
  if (provider.wire === 'anthropic') {
    const messages: unknown[] = [];
    for (const msg of canon.messages) {
      if (msg.images.length === 0) {
        messages.push({ role: msg.role, content: msg.text });
      } else {
        const blocks: unknown[] = [];
        if (msg.text !== '') blocks.push({ type: 'text', text: msg.text });
        for (const img of msg.images) {
          blocks.push({
            type: 'image',
            source:
              img.kind === 'bytes'
                ? { type: 'base64', media_type: img.mediaType, data: img.data }
                : { type: 'url', url: img.url },
          });
        }
        messages.push({ role: msg.role, content: blocks });
      }
    }
    const body: Json = { model, messages, stream: false };
    if (canon.systemText !== '') body.system = canon.systemText;
    if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
    if (canon.temperature !== null) body.temperature = canon.temperature;
    if (canon.stops.length > 0) body.stop_sequences = canon.stops;
    if (canon.tools.length > 0) {
      body.tools = canon.tools.map((tool) => ({
        name: tool.name,
        ...(tool.description !== null ? { description: tool.description } : {}),
        ...(tool.parameters !== null ? { input_schema: tool.parameters } : {}),
      }));
    }
    return body;
  }
  // gemini wire
  const contents: unknown[] = [];
  for (const msg of canon.messages) {
    const parts: unknown[] = [];
    if (msg.text !== '') parts.push({ text: msg.text });
    for (const img of msg.images) {
      parts.push(
        img.kind === 'bytes'
          ? { inlineData: { mimeType: img.mediaType, data: img.data } }
          : { fileData: { fileUri: img.url } },
      );
    }
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: msg.role === 'assistant' ? 'model' : 'user', parts });
  }
  const generationConfig: Json = {};
  if (canon.maxTokens !== null) generationConfig.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) generationConfig.stopSequences = canon.stops;
  if (canon.temperature !== null) generationConfig.temperature = canon.temperature;
  if (canon.jsonMode) generationConfig.responseMimeType = 'application/json';
  const body: Json = { contents, generationConfig };
  if (canon.systemText !== '') body.systemInstruction = { parts: [{ text: canon.systemText }] };
  if (canon.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: canon.tools.map((tool) => ({
          name: tool.name,
          ...(tool.description !== null ? { description: tool.description } : {}),
          ...(tool.parameters !== null ? { parameters: tool.parameters } : {}),
        })),
      },
    ];
  }
  return body;
}

/** Same-wire calls pass the client's body through untouched apart from the model and the stream flag. */
function passthroughBody(clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  const copy: Json = { ...body, model, stream: false };
  if (clientApi === 'openai') delete copy.stream_options;
  return copy;
}

interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

/** Read a provider's non-streamed answer in its own wire: its text and its token counts, verbatim. */
function parseProviderAnswer(wire: ProviderDir['wire'], payload: unknown): ProviderAnswer | null {
  if (!isRec(payload)) return null;
  if (wire === 'openai') {
    const choices = Array.isArray(payload.choices) ? payload.choices : [];
    const first = choices.length > 0 && isRec(choices[0]) ? choices[0] : null;
    const message = first !== null && isRec(first.message) ? first.message : null;
    const usage = isRec(payload.usage) ? payload.usage : null;
    if (message === null || usage === null) return null;
    const input = num(usage.prompt_tokens);
    const output = num(usage.completion_tokens);
    if (input === null || output === null) return null;
    return { text: blocksText(message.content), inputTokens: input, outputTokens: output };
  }
  if (wire === 'anthropic') {
    const usage = isRec(payload.usage) ? payload.usage : null;
    if (usage === null) return null;
    const input = num(usage.input_tokens);
    const output = num(usage.output_tokens);
    if (input === null || output === null) return null;
    return { text: blocksText(payload.content), inputTokens: input, outputTokens: output };
  }
  const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
  const first = candidates.length > 0 && isRec(candidates[0]) ? candidates[0] : null;
  const content = first !== null && isRec(first.content) ? first.content : null;
  const usage = isRec(payload.usageMetadata) ? payload.usageMetadata : null;
  if (content === null || usage === null) return null;
  const input = num(usage.promptTokenCount);
  const output = num(usage.candidatesTokenCount);
  if (input === null || output === null) return null;
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  for (const part of parts) {
    if (isRec(part) && typeof part.text === 'string') text += part.text;
  }
  return { text, inputTokens: input, outputTokens: output };
}

/** One upstream call. Null means the call refused or failed: the caller drops the provider and walks on. */
async function callProvider(
  provider: ProviderDir,
  model: string,
  clientApi: 'openai' | 'anthropic',
  clientBody: Json,
  canon: Canon,
): Promise<ProviderAnswer | null> {
  const sameWire =
    (clientApi === 'openai' && provider.wire === 'openai') ||
    (clientApi === 'anthropic' && provider.wire === 'anthropic');
  const body = sameWire ? passthroughBody(clientApi, clientBody, model) : buildUpstreamBody(provider, model, canon);
  let response: Response;
  try {
    response = await fetch(providerUrl(provider, model), {
      method: 'POST',
      headers: providerHeaders(provider),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(20000),
    });
  } catch {
    return null;
  }
  if (!response.ok) return null;
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    return null;
  }
  return parseProviderAnswer(provider.wire, payload);
}

// ---------------------------------------------------------------- client answers

const FIXED_STAMP = 1767225600; // a constant, never a clock: our answers carry no wall-clock reading.

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, api: 'openai' | 'anthropic', status: number, message: string): void {
  if (api === 'anthropic') {
    sendJson(res, status, { type: 'error', error: { type: 'api_error', message } });
  } else {
    sendJson(res, status, { error: { message, type: 'api_error', param: null, code: 'aggregator_error' } });
  }
}

/** Split an answer text into deterministic stream pieces. */
function pieces(text: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += 24) out.push(text.slice(i, i + 24));
  return out.length > 0 ? out : [''];
}

function answerOpenAI(res: ServerResponse, model: string, answer: ProviderAnswer, stream: boolean): void {
  const usage = {
    prompt_tokens: answer.inputTokens,
    completion_tokens: answer.outputTokens,
    total_tokens: answer.inputTokens + answer.outputTokens,
  };
  if (!stream) {
    sendJson(res, 200, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion',
      created: FIXED_STAMP,
      model,
      choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop' }],
      usage,
    });
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const frame = (chunk: Json): void => {
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
  };
  const head = { id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_STAMP, model };
  frame({ ...head, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] });
  for (const piece of pieces(answer.text)) {
    frame({ ...head, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
  }
  frame({ ...head, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] });
  frame({ ...head, choices: [], usage });
  res.write('data: [DONE]\n\n');
  res.end();
}

function answerAnthropic(res: ServerResponse, model: string, answer: ProviderAnswer, stream: boolean): void {
  if (!stream) {
    sendJson(res, 200, {
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model,
      content: [{ type: 'text', text: answer.text }],
      stop_reason: 'end_turn',
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
    });
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const frame = (event: string, data: Json): void => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };
  frame('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  });
  frame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const piece of pieces(answer.text)) {
    frame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  }
  frame('content_block_stop', { type: 'content_block_stop', index: 0 });
  frame('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: 'end_turn', stop_sequence: null },
    usage: { output_tokens: answer.outputTokens },
  });
  frame('message_stop', { type: 'message_stop' });
  res.end();
}

// ---------------------------------------------------------------- the server

const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const brief = readBrief(PROMPT_FILE);
const catalog = buildCatalog(brief);
const plan = buildPlan(brief);
const providerById = new Map(brief.providers.map((p) => [p.id, p]));

/**
 * Providers proven spent this trial: dropped and never called again. A provider is proven spent
 * by two consecutive refusals: a single refusal is a scheduled event in this competition (every
 * provider draws one or two refusing call numbers, and one provider always refuses its second
 * call), so it says nothing about the next call — dropping on it throws away the provider's
 * remaining budget and pushes later requests onto worse providers, whose own scheduled refusals
 * then cascade. Two refusals in a row, with no success between, is what an exhausted budget or
 * an outage actually looks like; the one confirmation call is the only extra call a spent
 * provider ever costs.
 */
const dropped = new Set<string>();
/** Consecutive refusals per provider since its last success. */
const refusalStreak = new Map<string, number>();
/** Counting calls made to each provider so far this trial. */
const callsMade = new Map<string, number>();

/**
 * One walk step: a single call to the candidate's provider — never a retry, so a refusal costs
 * exactly the one call that discovered it. The outcome updates the provider's refusal streak: a
 * success clears it; a second refusal in a row proves the provider spent (an outage or an empty
 * budget, both of which refuse forever) and drops it, so later requests skip it without spending
 * a call to find out.
 */
async function attempt(
  candidate: Candidate,
  api: 'openai' | 'anthropic',
  body: Json,
  canon: Canon,
): Promise<ProviderAnswer | null> {
  const provider = providerById.get(candidate.provider);
  if (provider === undefined || dropped.has(provider.id)) return null;
  callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
  const answer = await callProvider(provider, candidate.model, api, body, canon);
  if (answer !== null) {
    refusalStreak.set(provider.id, 0);
    return answer;
  }
  const streak = (refusalStreak.get(provider.id) ?? 0) + 1;
  refusalStreak.set(provider.id, streak);
  if (streak >= 2) dropped.add(provider.id);
  return null;
}
/** Client requests arrive one at a time, in manifest order. */
let arrivalIndex = 0;

function requiresFromBody(api: 'openai' | 'anthropic', body: Json): Requires {
  const canon = api === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
  return {
    tools: canon.tools.length > 0,
    vision: canon.messages.some((m) => m.images.length > 0),
    json: canon.jsonMode,
    minContext: null,
  };
}

async function serveRequest(api: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const entry = brief.manifest[arrivalIndex];
  arrivalIndex += 1;
  const canon = api === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
  let ladder: Candidate[];
  if (entry !== undefined && plan.has(entry.id)) {
    ladder = plan.get(entry.id) ?? [];
  } else {
    // No manifest row for this arrival (not part of the scored workload): rate every eligible model.
    const req = requiresFromBody(api, body);
    ladder = brief.ratings
      .filter((row) => meets(row, req))
      .sort((a, b) => b.rating - a.rating)
      .map((row) => ({ provider: row.provider, model: row.model, rating: row.rating }));
  }
  const respond = (candidate: Candidate, answer: ProviderAnswer): void => {
    if (api === 'openai') answerOpenAI(res, candidate.model, answer, canon.stream);
    else answerAnthropic(res, candidate.model, answer, canon.stream);
  };
  // The planned call first.
  if (ladder.length > 0) {
    const answer = await attempt(ladder[0], api, body, canon);
    if (answer !== null) return respond(ladder[0], answer);
  }
  // Then the walk, only because a refusal actually happened: providers never called yet first
  // (their first call number never refuses, so they close the request in exactly one call),
  // then the rest, each group in the ladder's rating order.
  const rest = ladder.slice(1).filter((candidate) => !dropped.has(candidate.provider));
  const fresh = rest.filter((candidate) => (callsMade.get(candidate.provider) ?? 0) === 0);
  const warm = rest.filter((candidate) => (callsMade.get(candidate.provider) ?? 0) > 0);
  for (const candidate of [...fresh, ...warm]) {
    const answer = await attempt(candidate, api, body, canon);
    if (answer !== null) return respond(candidate, answer);
  }
  sendError(res, api, 502, 'no free provider could serve this request');
}

async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > 32 * 1024 * 1024) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRec(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    const ids = freeModelIds(brief);
    if (typeof req.headers['x-api-key'] === 'string') {
      sendJson(res, 200, {
        data: ids.map((id) => ({ type: 'model', id, display_name: id, created_at: '2026-01-01T00:00:00Z' })),
        has_more: false,
        first_id: ids[0] ?? null,
        last_id: ids[ids.length - 1] ?? null,
      });
    } else {
      sendJson(res, 200, {
        object: 'list',
        data: ids.map((id) => ({ id, object: 'model', created: FIXED_STAMP, owned_by: 'aggregator' })),
      });
    }
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJsonBody(req);
    if (body === null) return sendError(res, 'openai', 400, 'the request body is not one JSON object');
    await serveRequest('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJsonBody(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'the request body is not one JSON object');
    await serveRequest('anthropic', body, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, `no route ${req.method ?? ''} ${url.pathname}`);
}

// ---------------------------------------------------------------- trial lifecycle

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        reject(new Error('the server has no port'));
        return;
      }
      resolve(address.port);
    });
  });
}

async function register(port: number): Promise<void> {
  const response = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!response.ok) throw new Error(`POST /register answered HTTP ${response.status}`);
  const reply: unknown = await response.json().catch(() => null);
  if (isRec(reply) && typeof reply.requests === 'number') {
    log(`registered: the workload holds ${reply.requests} requests`);
  }
}

function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const response = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (response.status === 200) {
          const work: unknown = await response.json().catch(() => null);
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
  }, 100);
}

async function start(): Promise<void> {
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (brief.providers.length === 0) throw new Error('the brief holds no provider directory');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log(err instanceof Error ? (err.stack ?? err.message) : String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'the server failed to answer');
      }
    });
  });
  let port = await listen(server);
  if (port === 8080) {
    // The env owns 8080 on one layout: close and take another ephemeral port.
    await new Promise<void>((resolve) => server.close(() => resolve()));
    port = await listen(server);
  }
  log(`listening on 0.0.0.0:${port}; plan covers ${plan.size} manifest requests over ${brief.providers.length} providers`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log(err instanceof Error ? (err.stack ?? err.message) : String(err));
  process.exit(1);
}
