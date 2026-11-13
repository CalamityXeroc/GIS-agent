import { readFileSync } from 'node:fs'
import { fileURLToPath, URL } from 'node:url'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

/** 版本号只有一处真相：读 pyproject.toml，顶栏 badge 直接用。 */
function readAppVersion(): string {
  try {
    const pyproject = readFileSync(
      fileURLToPath(new URL('../pyproject.toml', import.meta.url)),
      'utf8'
    )
    return /^version\s*=\s*"([^"]+)"/m.exec(pyproject)?.[1] ?? '0.0.0'
  } catch {
    return '0.0.0'
  }
}

export default defineConfig({
  // 产物由 FastAPI 的 StaticFiles 挂在 "/" 下。
  // 这里必须用绝对 base：SPA 子路由（/runs/demo-01）刷新时，相对路径
  // ``./assets/…`` 会被解析成 ``/runs/assets/…`` 直接 404（白屏）；
  // 服务固定挂根路径，所以 ``/assets/…`` 才是正确写法。
  base: '/',
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  define: {
    __APP_VERSION__: JSON.stringify(readAppVersion()),
  },
  build: {
    // 构建产物直接落到 Python 包内，`pip install` 完即可 `gis-agent web`。
    outDir: fileURLToPath(new URL('../src/gis_cli/web/static', import.meta.url)),
    emptyOutDir: true,
    target: 'es2022',
    // CSS 保持 OKLCH 原值：低版本目标会被降级成 hex/lab，token 就失真了。
    cssTarget: 'chrome111',
    chunkSizeWarningLimit: 1200,
  },
})
