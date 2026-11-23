#!/usr/bin/env node
/**
 * DeepSeek Anthropic Proxy Daemon
 * 自动重启代理进程，确保 Claude Code 始终能连接
 */

const { spawn } = require('child_process');
const path = require('path');

const PROXY_SCRIPT = path.join(__dirname, 'deepseek_anthropic_proxy.js');
const RESTART_DELAY = 3000; // 崩溃后 3 秒重启

let proxyProcess = null;

function startProxy() {
  console.log(`[DAEMON] Starting proxy at ${new Date().toISOString()}`);
  proxyProcess = spawn('node', [PROXY_SCRIPT], {
    stdio: ['inherit', 'inherit', 'inherit'],
    detached: false,
  });

  proxyProcess.on('exit', (code, signal) => {
    console.log(`[DAEMON] Proxy exited with code ${code}, signal ${signal}`);
    console.log(`[DAEMON] Restarting in ${RESTART_DELAY}ms...`);
    setTimeout(startProxy, RESTART_DELAY);
  });

  proxyProcess.on('error', (error) => {
    console.log(`[DAEMON] Proxy error: ${error.message}`);
  });
}

process.on('SIGTERM', () => {
  console.log('[DAEMON] Received SIGTERM, shutting down gracefully...');
  if (proxyProcess) {
    proxyProcess.kill();
  }
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('[DAEMON] Received SIGINT, shutting down gracefully...');
  if (proxyProcess) {
    proxyProcess.kill();
  }
  process.exit(0);
});

console.log('[DAEMON] DeepSeek Anthropic Proxy Daemon started');
startProxy();
