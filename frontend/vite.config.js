import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // In local dev, the frontend (Vite, :5173) and backend (FastAPI, :8000) run
    // as separate processes. This proxies relative API calls to the backend so
    // the app code can use the same "/predict" path locally and in production
    // (where FastAPI serves the built frontend itself, same origin).
    proxy: {
      '/predict': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
})
