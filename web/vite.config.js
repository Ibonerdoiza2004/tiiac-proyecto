import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// En desarrollo, Vite sirve la interfaz y reenvía /api (HTTP y WebSocket) al backend
// FastAPI. En producción no hay proxy: FastAPI sirve la carpeta dist/ que genera
// `npm run build`, así que la interfaz y la API comparten origen.
const API_DESARROLLO = process.env.TIIAC_API_URL ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': { target: API_DESARROLLO, changeOrigin: true, ws: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/pruebas/preparar.js'],
    include: ['src/**/*.test.{js,jsx}'],
    restoreMocks: true,
  },
});
