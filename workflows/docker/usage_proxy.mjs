// Trial-local ArtifactNet relay: fixed upstreams, no streaming or prompt logging.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const config = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
for (const role of ['llm', 'embedding']) {
  const name = config.upstreams[role].api_key_env;
  if (!process.env[name]) throw new Error(`Missing API key environment variable: ${name}`);
}
fs.mkdirSync(path.dirname(config.usage_file), { recursive: true });
fs.closeSync(fs.openSync(config.usage_file, 'a'));
const incoming = new Set();
const active = new Set();
let stopping = false;

function cleanUsage(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const result = {};
  for (const [key, item] of Object.entries(value)) {
    if (!/(^|_)(tokens|cost)(_|$)/.test(key)) continue;
    if (typeof item === 'number' && Number.isFinite(item)) result[key] = item;
    else if (item && typeof item === 'object' && !Array.isArray(item)) result[key] = cleanUsage(item);
  }
  return result;
}

const server = http.createServer(async (req, res) => {
  if (stopping) { res.writeHead(503).end(); return; }
  if (req.method === 'GET' && req.url === '/health') { res.writeHead(200).end('ok'); return; }
  const route = /^\/(?:m\/([A-Za-z0-9][A-Za-z0-9_.-]*)|unassigned)\/(llm|embedding)\/v1\/(chat\/completions|embeddings)$/.exec(req.url);
  if (req.method !== 'POST' || !route) { res.writeHead(404).end(); return; }
  const [, milestone, role, operation] = route;
  if (operation !== (role === 'llm' ? 'chat/completions' : 'embeddings')) { res.writeHead(404).end(); return; }
  const target = config.upstreams[role];
  const started = Date.now();
  let body;
  incoming.add(req);
  try {
    const chunks = [];
    let size = 0;
    for await (const chunk of req) {
      size += chunk.length;
      if (size > 16 * 1024 * 1024) { res.writeHead(413).end(); return; }
      chunks.push(chunk);
    }
    body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    res.writeHead(400).end(); return;
  } finally {
    incoming.delete(req);
  }
  if (!body || Array.isArray(body) || body.model !== target.model || (body.stream !== undefined && body.stream !== false)) {
    res.writeHead(400).end(); return;
  }
  if (stopping) { res.writeHead(503).end(); return; }
  const controller = new AbortController();
  const timeout = AbortSignal.timeout(config.request_timeout_ms ?? 120000);
  active.add(controller);
  let status = 502, usage = null, error = null, responseBody;
  try {
    const upstream = await fetch(`${target.base_url.replace(/\/$/, '')}/${operation}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${process.env[target.api_key_env]}` },
      body: JSON.stringify(body), redirect: 'error',
      signal: AbortSignal.any([controller.signal, timeout]),
    });
    responseBody = await upstream.text();
    status = upstream.status;
    try { usage = cleanUsage(JSON.parse(responseBody).usage); } catch { /* missing provider usage */ }
  } catch {
    error = controller.signal.aborted ? 'shutdown' : timeout.aborted ? 'timeout' : 'upstream_error';
    status = error === 'timeout' ? 504 : 502;
    responseBody = JSON.stringify({ error: 'workflow_api_request_failed' });
  } finally {
    active.delete(controller);
  }
  const record = {
    schema_version: 1, timestamp: new Date(started).toISOString(),
    trial_id: config.trial_id, trial_name: config.trial_name, milestone: milestone ?? null,
    role, model: target.model, status, duration_ms: Date.now() - started,
    usage, cost_usd: typeof usage?.cost === 'number' ? usage.cost : null, error,
  };
  try {
    fs.appendFileSync(config.usage_file, JSON.stringify(record) + '\n');
  } catch {
    console.error('Workflow usage log append failed; stopping the proxy');
    process.exit(1);
  }
  res.writeHead(status, { 'content-type': 'application/json' }).end(responseBody);
});

process.on('SIGTERM', () => {
  stopping = true;
  server.close();
  for (const req of incoming) req.destroy();
  for (const controller of active) controller.abort();
});
server.listen(config.port ?? 8080, '0.0.0.0');
