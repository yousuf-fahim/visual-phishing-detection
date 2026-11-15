// Parity test: does the JavaScript preprocessing + ONNX model reproduce Python exactly?
//
// Run from web/:  npm test     (after: .venv/Scripts/python -m src.export_onnx)
//
// Fixtures are written by src/export_onnx.py into data/processed/js_parity/ (git-ignored; some
// are real dataset screenshots, so they never go into web/). For each image:
//   <id>_rgb.bin   decoded RGB pixels (what the browser's PNG decoder would also produce)
//   <id>_224.bin   Python's 224x224 letterboxed pixels (the exact training input)
//   fixtures.json  size and Python's ONNX probability
// Pass criteria: pixel max |diff| <= 1 and probability |diff| < 1e-3 for every image.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import * as ort from "onnxruntime-web";
import { letterbox, toModelInput, SIZE } from "../src/preprocess.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const FIX = path.resolve(here, "../../data/processed/js_parity");
const MODEL = path.resolve(here, "../public/image_model.onnx");

const fixtures = JSON.parse(readFileSync(path.join(FIX, "fixtures.json"), "utf8"));
const session = await ort.InferenceSession.create(readFileSync(MODEL));

let worstPixel = 0, worstProb = 0, pixelsOff = 0, failures = 0;
for (const f of fixtures) {
  const id = String(f.id).padStart(2, "0");
  const rgb = new Uint8Array(readFileSync(path.join(FIX, `${id}_rgb.bin`)));
  const expected = new Uint8Array(readFileSync(path.join(FIX, `${id}_224.bin`)));
  const t0 = performance.now();
  const ours = letterbox(rgb, f.width, f.height);
  const ms = performance.now() - t0;

  let maxDiff = 0, nDiff = 0;
  for (let i = 0; i < ours.length; i++) {
    const d = Math.abs(ours[i] - expected[i]);
    if (d > 0) nDiff++;
    if (d > maxDiff) maxDiff = d;
  }
  const input = new ort.Tensor("float32", toModelInput(ours), [1, 3, SIZE, SIZE]);
  const p = (await session.run({ image: input })).p_phish.data[0];
  const dp = Math.abs(p - f.p_phish);
  const ok = maxDiff <= 1 && dp < 1e-3;
  if (!ok) failures++;
  worstPixel = Math.max(worstPixel, maxDiff);
  worstProb = Math.max(worstProb, dp);
  pixelsOff += nDiff;
  console.log(`${ok ? "PASS" : "FAIL"} #${id} ${f.width}x${f.height}  max|dpixel|=${maxDiff} ` +
              `(${nDiff} values differ)  p_js=${p.toFixed(6)} p_py=${f.p_phish.toFixed(6)}  ` +
              `|dp|=${dp.toExponential(2)}  resize ${ms.toFixed(0)} ms`);
}
console.log(`\n${fixtures.length - failures}/${fixtures.length} passed. worst pixel diff ${worstPixel}, ` +
            `values differing in total ${pixelsOff}, worst |dp| ${worstProb.toExponential(2)}`);
process.exit(failures ? 1 : 0);
