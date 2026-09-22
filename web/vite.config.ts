import { defineConfig } from 'vite'
import preact from '@preact/preset-vite'

// web 服务独立于后端: 开发期把 /v1 代理到引擎地址, 地址只从环境变量来, 不写死进业务代码
const backend = process.env.JEV_BACKEND ?? 'http://127.0.0.1:8000'

export default defineConfig({
  base: './',
  plugins: [preact()],
  server: {
    host: '127.0.0.1',
    port: 5273,
    strictPort: true,
    proxy: { '/v1': { target: backend, changeOrigin: true } },
  },
  build: { target: 'es2021', outDir: 'dist', emptyOutDir: true },
})
