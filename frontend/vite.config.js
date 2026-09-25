import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Forward API calls to the Python server (server/app.py) so there is no CORS setup.
  server: {
    proxy: { "/api": "http://127.0.0.1:5000" },
  },
});
