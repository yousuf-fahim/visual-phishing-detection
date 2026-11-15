// Demo app: decode a dropped screenshot locally, preprocess exactly like training
// (preprocess.js), and run the ONNX model with onnxruntime-web (WebAssembly, self-hosted).
// Nothing is uploaded: the page's Content-Security-Policy only allows same-origin requests.
import { letterbox, toModelInput, toRGB, SIZE } from "./preprocess.js";

const $ = (id) => document.getElementById(id);
const status = (msg) => { $("status").textContent = msg; };

let session = null;
let ort = null;
let card = {};

async function init() {
  try {
    card = await (await fetch("./model_card.json")).json();
    fillCard(card);
    ort = await import("./ort/ort.wasm.min.mjs");
    // Absolute URL, resolved against this page (a bare "./ort/" would be resolved against ort's own folder).
    ort.env.wasm.wasmPaths = new URL("./ort/", location.href).href;
    ort.env.wasm.numThreads = 1; // static hosts aren't cross-origin isolated; one thread is enough
    session = await ort.InferenceSession.create("./image_model.onnx", { executionProviders: ["wasm"] });
    status("Model ready. Drop a screenshot.");
    if (new URLSearchParams(location.search).has("selftest")) await selfTest();
  } catch (err) {
    status(`Could not load the model: ${err.message}`);
    console.error(err);
  }
}

function fillCard(c) {
  document.querySelectorAll("[data-card]").forEach((el) => {
    const v = c[el.dataset.card];
    if (v === undefined) return;
    if (el.hasAttribute("data-pct")) el.textContent = `${Math.round(v * 100)}%`;
    else el.textContent = typeof v === "number" ? v.toFixed(3) : String(v);
  });
}

// Decode with the browser's own decoder. colorSpaceConversion "none" matches Pillow, which ignores
// embedded colour profiles. Transparent pixels are an exception: canvas stores premultiplied alpha
// (screenshots are normally opaque, so this rarely matters).
async function decode(blob) {
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const { data } = ctx.getImageData(0, 0, bmp.width, bmp.height);
  const out = { rgb: toRGB(data, bmp.width, bmp.height, 4), width: bmp.width, height: bmp.height };
  bmp.close();
  return out;
}

async function score(blob) {
  const { rgb, width, height } = await decode(blob);
  const img224 = letterbox(rgb, width, height);
  const input = new ort.Tensor("float32", toModelInput(img224), [1, 3, SIZE, SIZE]);
  const p = (await session.run({ image: input })).p_phish.data[0];
  return { p, width, height };
}

async function handle(file) {
  if (!session) return;
  if (!file || !file.type.startsWith("image/")) {
    status("Please choose an image file (PNG, JPEG or WebP).");
    return;
  }
  status("Scoring…");
  try {
    const url = URL.createObjectURL(file);
    $("preview").src = url;
    const { p, width, height } = await score(file);
    show(p, width, height);
    status(`Scored a ${width}×${height} image locally.`);
  } catch (err) {
    status(`Could not read that image: ${err.message}`);
  }
}

function show(p, width, height) {
  $("result").hidden = false;
  const pct = Math.round(p * 1000) / 10;
  $("score").textContent = `${pct.toFixed(1)}%`;
  $("meter-fill").style.width = `${Math.max(1, pct)}%`;
  $("meter-fill").className = p >= 0.5 ? "high" : "low";
  $("verdict").textContent = p >= 0.5
    ? "Looks visually similar to the phishing pages in the training data. That alone does not mean this site is malicious; check the address bar."
    : "Looks more like the legitimate pages in the training data. That alone does not mean this site is safe.";
  const aspect = width / height;
  const warn = $("aspect-warning");
  warn.hidden = Math.abs(aspect - 4 / 3) < 0.25;
  warn.textContent = `This image is ${width}×${height}. The model was trained on 1280×960 page screenshots, ` +
    "so very tall, very wide or cropped images are less reliable.";
}

// ?selftest=1: run the bundled synthetic image (not a real site) and print the result, so the
// whole browser pipeline (decoder + preprocessing + wasm model) can be checked against Python.
async function selfTest() {
  const blob = await (await fetch("./selftest.png")).blob();
  const { p } = await score(blob);
  const expected = card.selftest_expected_p_phish;
  const ok = Math.abs(p - expected) < 1e-3;
  const box = $("selftest");
  box.hidden = false;
  box.textContent = `SELFTEST ${ok ? "PASS" : "FAIL"} p=${p.toFixed(6)} expected=${expected}`;
}

// ---- drag and drop / file picker ----
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => {
  e.preventDefault();
  drop.classList.add("over");
}));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => {
  e.preventDefault();
  drop.classList.remove("over");
}));
drop.addEventListener("drop", (e) => handle(e.dataTransfer.files[0]));
$("file").addEventListener("change", (e) => handle(e.target.files[0]));
document.addEventListener("paste", (e) => {
  const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
  if (item) handle(item.getAsFile());
});

init();
