// Screenshot preprocessing, matching src/preprocess_images.py exactly.
//
// Python resizes with Pillow's LANCZOS filter. A browser <canvas> uses its own
// resampling, which would feed the model slightly different pixels than it saw in
// training. So this file is a direct port of Pillow's resampling code
// (libImaging/Resample.c): separable Lanczos-3, horizontal pass first, then vertical,
// 22-bit fixed-point weights, and 8-bit rounding between the passes.
// web/test/parity.mjs checks it against the Python arrays.
//
// Works in the browser and in Node (no DOM needed): input is raw RGB(A) pixels.

export const SIZE = 224;
export const PAD_RGB = [124, 116, 104]; // ImageNet mean * 255, same as Python

const PRECISION_BITS = 32 - 8 - 2; // 22, as in Pillow
const ONE = 2 ** PRECISION_BITS;

function sinc(x) {
  if (x === 0) return 1.0;
  x *= Math.PI;
  return Math.sin(x) / x;
}

function lanczos(x) {
  return x >= -3.0 && x < 3.0 ? sinc(x) * sinc(x / 3) : 0.0;
}

// Pillow precompute_coeffs + normalize_coeffs_8bpc for one axis.
function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1.0);
  const support = 3.0 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const w = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1.0 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0.0;
    for (let x = 0; x < xmax; x++) {
      w[x] = lanczos((x + xmin - center + 0.5) * ss);
      ww += w[x];
    }
    for (let x = 0; x < ksize; x++) {
      const v = x < xmax && ww !== 0.0 ? w[x] / ww : x < xmax ? w[x] : 0.0;
      // C's (int) cast truncates toward zero
      kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk };
}

function clip8(v) {
  if (v >= ONE * 256) return 255;
  if (v <= 0) return 0;
  return Math.floor(v / ONE);
}

// Resize an RGB image (Uint8Array, 3 bytes/pixel) from (w, h) to (outW, outH), like
// PIL Image.resize((outW, outH), Image.Resampling.LANCZOS).
export function resizeLanczos(rgb, w, h, outW, outH) {
  const H = coefficients(w, outW);
  const V = coefficients(h, outH);
  // Rows of the source actually used by the vertical pass (Pillow does the same).
  const yFirst = V.bounds[0];
  const yLast = V.bounds[outH * 2 - 2] + V.bounds[outH * 2 - 1];
  const needH = outW !== w;
  const needV = outH !== h;

  let src = rgb, srcW = w, rowOffset = 0, tmpH = h;
  if (needH) {
    tmpH = yLast - yFirst;
    const tmp = new Uint8Array(outW * tmpH * 3);
    for (let y = 0; y < tmpH; y++) {
      const inRow = (y + yFirst) * w * 3;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = H.bounds[xx * 2], xmax = H.bounds[xx * 2 + 1], k = xx * H.ksize;
        for (let c = 0; c < 3; c++) {
          let ss = ONE / 2;
          for (let x = 0; x < xmax; x++) ss += rgb[inRow + (x + xmin) * 3 + c] * H.kk[k + x];
          tmp[(y * outW + xx) * 3 + c] = clip8(ss);
        }
      }
    }
    src = tmp;
    srcW = outW;
    rowOffset = yFirst; // vertical bounds are relative to the temp image
  }
  if (!needV) return src;
  const out = new Uint8Array(outW * outH * 3);
  for (let yy = 0; yy < outH; yy++) {
    const ymin = V.bounds[yy * 2] - (needH ? rowOffset : 0), ymax = V.bounds[yy * 2 + 1], k = yy * V.ksize;
    for (let xx = 0; xx < outW; xx++) {
      for (let c = 0; c < 3; c++) {
        let ss = ONE / 2;
        for (let y = 0; y < ymax; y++) ss += src[((y + ymin) * srcW + xx) * 3 + c] * V.kk[k + y];
        out[(yy * outW + xx) * 3 + c] = clip8(ss);
      }
    }
  }
  return out;
}

// Python's round(): halves go to the nearest even integer (round(2.5) == 2).
export function pyRound(x) {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// RGBA (from canvas/ImageData) or RGB -> RGB. Alpha is dropped, like PIL convert("RGB").
export function toRGB(pixels, w, h, channels) {
  if (channels === 3) return pixels;
  const out = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < w * h * 4; i += 4, j += 3) {
    out[j] = pixels[i];
    out[j + 1] = pixels[i + 1];
    out[j + 2] = pixels[i + 2];
  }
  return out;
}

// Letterbox exactly like Python's letterbox(): keep aspect, Lanczos resize, centre on grey.
// Returns a 224x224x3 Uint8Array.
export function letterbox(rgb, w, h) {
  const scale = SIZE / Math.max(w, h);
  const newW = Math.max(1, pyRound(w * scale));
  const newH = Math.max(1, pyRound(h * scale));
  const resized = resizeLanczos(rgb, w, h, newW, newH);
  const canvas = new Uint8Array(SIZE * SIZE * 3);
  for (let i = 0; i < SIZE * SIZE; i++) canvas.set(PAD_RGB, i * 3);
  const left = Math.floor((SIZE - newW) / 2), top = Math.floor((SIZE - newH) / 2);
  for (let y = 0; y < newH; y++) {
    canvas.set(resized.subarray(y * newW * 3, (y + 1) * newW * 3), ((y + top) * SIZE + left) * 3);
  }
  return canvas;
}

// 224x224x3 uint8 (HWC) -> Float32Array NCHW in 0..255. The ONNX model does /255 and
// mean/std normalisation itself (see src/export_onnx.py), so nothing else happens here.
export function toModelInput(img224) {
  const out = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    out[i] = img224[i * 3];
    out[SIZE * SIZE + i] = img224[i * 3 + 1];
    out[2 * SIZE * SIZE + i] = img224[i * 3 + 2];
  }
  return out;
}
