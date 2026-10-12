import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The SPA is served by FastAPI from the same origin in production (one container,
// one port), so the built assets use relative URLs. In development Vite proxies
// /api to the local uvicorn process.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
  },
});
