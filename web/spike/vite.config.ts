import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// 独立 spike：端口写死，避免和主干 `web`（5173）抢。
export default defineConfig({
  plugins: [react()],
  server: { port: 4318, strictPort: true },
  preview: { port: 4318, strictPort: true },
  base: './',
})
