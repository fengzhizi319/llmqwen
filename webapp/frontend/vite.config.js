import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  publicDir: 'public',
  server: {
    port: 5173,
    proxy: {
      // 代理 API 请求到 Python 后端
      '/v1': {
        target: 'http://localhost:1235',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:1235',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
});
