const http = require('http');

// Simulate Claude Code's actual request with nested metadata object
const body = JSON.stringify({
  model: 'deepseek-v4-pro',
  max_tokens: 100,
  messages: [
    {
      role: 'user',
      content: 'Hello'
    }
  ],
  metadata: {
    device_id: 'test-device-123',
    account_uuid: 'acc-uuid-456',
    session_id: 'sess-id-789',
    extra_field: 'some_value'
  }
});

const options = {
  hostname: '127.0.0.1',
  port: 8787,
  path: '/messages',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
    'x-api-key': process.env.ANTHROPIC_API_KEY || 'sk-your-api-key-here',
    'anthropic-version': '2023-06-01'
  }
};

const req = http.request(options, (res) => {
  console.log(`[TEST] Response status: ${res.statusCode}`);
  let data = '';
  res.on('data', chunk => {
    data += chunk;
  });
  res.on('end', () => {
    console.log(`[TEST] Response length: ${data.length}`);
    if (data.length < 500) {
      console.log(`[TEST] Response preview: ${data.substring(0, 200)}`);
    }
    process.exit(0);
  });
});

req.on('error', (e) => {
  console.error(`[TEST] Request error: ${e.message}`);
  process.exit(1);
});

req.write(body);
req.end();

setTimeout(() => {
  console.error(`[TEST] Timeout - no response after 30s`);
  process.exit(1);
}, 30000);
