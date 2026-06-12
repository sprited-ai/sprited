import { defineConfig } from "vite";

// Demo site — deployed to https://sprited-ai.github.io/sprited by
// .github/workflows/pages.yml (which also drops the gallery assets from
// examples/ into the build output).
export default defineConfig({
  base: "/sprited/",
  build: { outDir: "dist", target: "esnext" },
});
