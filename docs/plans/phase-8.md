# Phase 8 plan: static browser demo

Constraints: pure static site, no backend, nothing uploaded anywhere (the screenshot never leaves the browser),
clearly labelled a research prototype. No Vercel. The build lands in web/dist and is deployed by Fahim.

## Model export (`src/export_onnx.py`)
- Fine-tuned MobileNetV3-small, full page, seed 42 (about 6 MB in fp32). It's exported with **normalisation baked in**:
  the ONNX input is the letterboxed 224x224 image as float 0-255 (NCHW), so JS only has to letterbox.
- Check: ONNX vs PyTorch on 20 validation images (max |Δp| < 1e-4).
- Quantisation: try onnxruntime dynamic int8 and keep it only if the size gain is worth it and val ROC-AUC holds (|Δ| < 0.002).
- Also writes `web/public/model_card.json`: every number shown in the UI, read from results/ files (no hand-typed numbers).

## Preprocessing parity (the hard part)
- Python resizes with PIL's Lanczos filter, while a browser canvas resizes differently. So `web/src/preprocess.js` is a
  direct port of PIL's resampling (separable Lanczos-3, horizontal pass then vertical, 22-bit fixed-point
  coefficients, uint8 between passes), followed by the same centred letterbox with the ImageNet-mean grey.
- Test (`web/test/parity.mjs`, Node + onnxruntime-web): for 20 validation screenshots, Python writes the
  decoded RGB pixels plus its 224x224 array and PyTorch probability to `data/processed/js_parity/` (git-ignored,
  never in web/). The test asserts pixel max |Δ| <= 1 and probability |Δ| < 1e-3. Nothing proceeds until it passes.
- Browser end-to-end check: a synthetic, non-phishing test image (drawn shapes) is bundled. `index.html?selftest=1`
  decodes it with the browser's own image decoder, runs the full pipeline, and prints the probability. Headless Edge
  compares this with Python's value for the same image.

## App (`web/`)
- Plain static HTML/CSS/JS, no bundler. `onnxruntime-web` comes from npm, and its wasm files are copied into the build (self-hosted).
- Drag-and-drop or file picker, shown locally. Phishing probability, a caveat banner, what the score means,
  and known limits (shortcut-hard 0.81, other datasets 0.91-0.93, non-English drop): numbers from model_card.json.
- No example gallery of phishing screenshots. An optional legitimate-site gallery is listed in DEPLOY.md for Fahim to capture.
- `web/build.mjs` copies everything into `web/dist` with relative paths. Tested with `python -m http.server`.
- `docs/DEPLOY.md`: exact steps for Cloudflare Pages and GitHub Pages.
