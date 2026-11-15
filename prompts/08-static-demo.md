# Phase 8: Static browser demo (90 min) OPTIONAL PLAN

Run in plan mode (Shift+Tab) after /clear, only if phases 0-5 and 7 are healthy. Do not use Vercel (account paused). Paste everything below the line.

Checkpoint: why is it safer to run the model in the browser than to upload the screenshot to a server?

---

Phase 8: a small demo. Plan first. Constraints: pure static site, no backend, nothing is uploaded anywhere, and it is clearly labeled a research prototype, not a security product.

1. Export the best image model to ONNX (models/image_model.onnx) and verify it matches the PyTorch outputs on 20 saved samples. Make the model small enough to load quickly in a browser (use the lightweight backbone, quantize if accuracy holds).
2. Build the app in web/ with the simplest stack (plain Vite or static HTML/JS), using onnxruntime-web, with model files self-hosted in the build.
3. Preprocessing in JS must match src/ exactly (pad to a 224x224 canvas, same normalization). Write a test comparing JS and Python outputs on the same images. Do not proceed until they match.
4. UI: drag and drop a screenshot, show the phishing probability with a clear caveat banner, a short explanation of what the score means, and the model's known limits from the report (language and time shift). Do not include any phishing screenshots as examples. If an example gallery is wanted, use screenshots of well-known legitimate sites that I capture myself, and tell me which to capture.
5. Build to web/dist with a relative base path, test with a local static server, and write exact deploy steps for Cloudflare Pages and GitHub Pages to docs/DEPLOY.md. I'll deploy myself.
6. Append the Phase 8 entry to docs/learning-log.md and commit.
