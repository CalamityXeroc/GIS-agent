const http = require('http');
const https = require('https');
const { URL } = require('url');

const LISTEN_HOST = process.env.DEEPSEEK_PROXY_HOST || '127.0.0.1';
const LISTEN_PORT = Number(process.env.DEEPSEEK_PROXY_PORT || '8787');
const TARGET_BASE = new URL(process.env.DEEPSEEK_TARGET_BASE || 'https://api.deepseek.com/anthropic');

function joinPathnames(basePath, requestPath) {
  const normalizedBase = basePath.endsWith('/') ? basePath.slice(0, -1) : basePath;
  const normalizedRequest = requestPath.startsWith('/') ? requestPath : `/${requestPath}`;
  if (normalizedBase === '') {
    return normalizedRequest;
  }

  return `${normalizedBase}${normalizedRequest}`;
}

function stripSensitiveMetadata(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return payload;
  }

  const cloned = { ...payload };
  if (cloned.metadata && typeof cloned.metadata === 'object') {
    const { user_id, ...rest } = cloned.metadata;
    if (user_id) {
      console.log(`[PROXY] Removed user_id from metadata`);
    }
    if (Object.keys(rest).length > 0) {
      cloned.metadata = rest;
    } else {
      delete cloned.metadata;
    }
  }

  return cloned;
}

function forwardRequest(clientReq, clientRes, rawBody) {
  const incomingUrl = new URL(clientReq.url || '/', 'http://localhost');
  const targetUrl = new URL(TARGET_BASE.toString());
  targetUrl.pathname = joinPathnames(TARGET_BASE.pathname, incomingUrl.pathname);
  targetUrl.search = incomingUrl.search;
  const headers = { ...clientReq.headers };
  delete headers.host;
  delete headers['content-length'];

  let bodyToSend = rawBody;
  const contentType = String(clientReq.headers['content-type'] || '').toLowerCase();
  if (contentType.includes('application/json') && rawBody.length > 0) {
    try {
      const parsed = JSON.parse(rawBody.toString('utf8'));
      const sanitized = stripSensitiveMetadata(parsed);
      bodyToSend = Buffer.from(JSON.stringify(sanitized));
      headers['content-length'] = Buffer.byteLength(bodyToSend);
    } catch {
      // Keep the original payload if parsing fails.
    }
  }

  const requestOptions = {
    protocol: targetUrl.protocol,
    hostname: targetUrl.hostname,
    port: targetUrl.port || (targetUrl.protocol === 'https:' ? 443 : 80),
    method: clientReq.method,
    path: `${targetUrl.pathname}${targetUrl.search}`,
    headers,
  };

  const transport = targetUrl.protocol === 'https:' ? https : http;
  let responseSent = false;

  function sendOnce(statusCode, responseHeaders, body) {
    if (responseSent || clientRes.writableEnded) {
      return;
    }
    responseSent = true;
    if (!clientRes.headersSent) {
      clientRes.writeHead(statusCode, responseHeaders);
    }
    clientRes.end(body);
  }

  const upstreamReq = transport.request(requestOptions, (upstreamRes) => {
    if (responseSent || clientRes.writableEnded) {
      upstreamRes.resume();
      return;
    }

    responseSent = true;
    const responseHeaders = { ...upstreamRes.headers, 'Connection': 'close' };
    if (!clientRes.headersSent) {
      clientRes.writeHead(upstreamRes.statusCode || 502, responseHeaders);
    }
    
    upstreamRes.on('error', (error) => {
      console.log(`[PROXY] Upstream response error: ${error.message}`);
      if (!clientRes.writableEnded) {
        clientRes.end();
      }
    });
    
    upstreamRes.pipe(clientRes, { end: true });
  });

  upstreamReq.on('error', (error) => {
    console.log(`[PROXY] Upstream request error: ${error.message}`);
    sendOnce(502, { 'content-type': 'application/json; charset=utf-8', 'Connection': 'close' }, JSON.stringify({ error: { message: error.message } }));
  });

  upstreamReq.on('timeout', () => {
    console.log(`[PROXY] Upstream request timeout`);
    upstreamReq.destroy();
    sendOnce(504, { 'content-type': 'application/json; charset=utf-8', 'Connection': 'close' }, JSON.stringify({ error: { message: 'Gateway Timeout' } }));
  });
  upstreamReq.setTimeout(30000);

  if (bodyToSend.length > 0 && clientReq.method && clientReq.method !== 'GET' && clientReq.method !== 'HEAD') {
    upstreamReq.write(bodyToSend);
  }
  upstreamReq.end();
}

const server = http.createServer((clientReq, clientRes) => {
  const chunks = [];
  clientReq.on('data', (chunk) => {
    chunks.push(chunk);
  });
  clientReq.on('error', (error) => {
    console.log(`[PROXY] Client error: ${error.message}`);
  });
  clientReq.on('end', () => {
    try {
      forwardRequest(clientReq, clientRes, Buffer.concat(chunks));
    } catch (error) {
      console.log(`[PROXY] Forward error: ${error.message}`);
      if (!clientRes.headersSent) {
        clientRes.writeHead(500, { 'content-type': 'application/json; charset=utf-8' });
      }
      if (!clientRes.writableEnded) {
        clientRes.end(JSON.stringify({ error: { message: error.message } }));
      }
    }
  });
});

server.on('error', (error) => {
  console.log(`[PROXY] Server error: ${error.message}`);
  if (error.code === 'EADDRINUSE') {
    console.log(`[PROXY] Port ${LISTEN_PORT} already in use, retrying in 5s...`);
    setTimeout(() => {
      server.close();
      server.listen(LISTEN_PORT, LISTEN_HOST, () => {
        console.log(`DeepSeek Anthropic proxy restarted on http://${LISTEN_HOST}:${LISTEN_PORT}`);
        console.log(`Forwarding to ${TARGET_BASE.origin}${TARGET_BASE.pathname}`);
      });
    }, 5000);
  }
});

server.listen(LISTEN_PORT, LISTEN_HOST, () => {
  console.log(`DeepSeek Anthropic proxy listening on http://${LISTEN_HOST}:${LISTEN_PORT}`);
  console.log(`Forwarding to ${TARGET_BASE.origin}${TARGET_BASE.pathname}`);
});
