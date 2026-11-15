// Build the static site into web/dist: copy the app, the model, and onnxruntime-web's
// WebAssembly files (self-hosted, no CDN). All paths are relative, so dist/ works from any
// sub-folder (GitHub Pages project site) or a domain root (Cloudflare Pages).
//
// Run from web/:  npm run build     (needs: npm install, and src/export_onnx.py has run)
import { cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.join(here, "dist");
const ortDir = path.join(here, "node_modules", "onnxruntime-web", "dist");

const required = ["public/image_model.onnx", "public/model_card.json", "public/selftest.png"];
for (const f of required) {
  if (!existsSync(path.join(here, f))) {
    console.error(`Missing ${f}. Run: .venv/Scripts/python -m src.export_onnx (from the repo root)`);
    process.exit(1);
  }
}

rmSync(dist, { recursive: true, force: true });
mkdirSync(path.join(dist, "ort"), { recursive: true });
for (const f of ["index.html", "app.js", "preprocess.js", "style.css"]) cpSync(path.join(here, "src", f), path.join(dist, f));
for (const f of ["image_model.onnx", "model_card.json", "selftest.png"]) cpSync(path.join(here, "public", f), path.join(dist, f));
// The wasm-only build of onnxruntime-web plus the runtime it loads.
for (const f of ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  cpSync(path.join(ortDir, f), path.join(dist, "ort", f));
}
// GitHub Pages: don't run Jekyll. Cloudflare Pages: correct type and caching for the model files.
writeFileSync(path.join(dist, ".nojekyll"), "");
writeFileSync(path.join(dist, "_headers"),
  "/*.wasm\n  Content-Type: application/wasm\n  Cache-Control: public, max-age=86400\n" +
  "/*.onnx\n  Cache-Control: public, max-age=86400\n");
console.log("Built web/dist");
