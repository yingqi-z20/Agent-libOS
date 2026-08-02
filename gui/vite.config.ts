import { configDefaults, defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

const guiRoot = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: false,
    fs: {
      strict: true,
      allow: [guiRoot]
    }
  },
  test: {
    exclude: [...configDefaults.exclude, "dist/**", "dist-electron/**", "e2e/**"]
  }
});
