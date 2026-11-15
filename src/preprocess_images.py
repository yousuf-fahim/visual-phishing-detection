"""Phase 3, step 1: letterbox every screenshot onto a fixed 224x224 canvas, once.

Run: .venv/Scripts/python -m src.preprocess_images

Why letterbox (resize keeping aspect ratio, then pad) instead of stretching: a
stretched 1280x960 page and a stretched 1920x1080 page would be distorted
differently, and the model could learn that distortion. Letterboxing makes every
input the same size with undistorted content. Padding uses the ImageNet mean
colour, which becomes exactly 0 after normalisation, so it carries no signal.

Phish-IRIS images are first cropped to a 4:3 viewport (see CROP_DATASETS below).

Two variants:
  full  the whole screenshot (Phish360 screenshots are a 1280x960 browser viewport)
  top   the top half only ("above the fold": logo, header, and usually the login box,
        which is what a user sees first and what a phishing page must imitate)

Output: data/processed/img224_{full,top}.npy  uint8 array (N, 224, 224, 3), rows in index.csv order.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

SIZE = 224
PAD_RGB = (124, 116, 104)  # ImageNet mean * 255
TOP_FRACTION = 0.5
# Phish-IRIS images come in hundreds of shapes, and shape predicts the label there
# (square 1024x1024 JPEGs are mostly 'other'). Letterboxing alone would still leak
# the shape through the width of the grey bands, so those images are first cropped
# to the 4:3 viewport shape that Phish360 screenshots already have (1280x960).
# Phish360 is left as is: its shape carries no signal (aspect-ratio AUC 0.50).
VIEWPORT_ASPECT = 4 / 3
CROP_DATASETS = {"phish-iris"}


def letterbox(im: Image.Image, size: int = SIZE) -> np.ndarray:
    w, h = im.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)  # antialiased downscale
    canvas = Image.new("RGB", (size, size), PAD_RGB)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))  # centred
    return np.asarray(canvas, dtype=np.uint8)


def crop_to_viewport(im: Image.Image, aspect: float = VIEWPORT_ASPECT) -> Image.Image:
    """Crop to a 4:3 browser viewport, keeping the TOP of the page (where logos and
    login boxes are). If the image is too wide, keep the horizontal centre."""
    w, h = im.size
    if w / h > aspect:  # too wide: trim the sides equally
        new_w = round(h * aspect)
        left = (w - new_w) // 2
        return im.crop((left, 0, left + new_w, h))
    new_h = round(w / aspect)  # too tall (or square): keep the top
    return im.crop((0, 0, w, min(h, new_h)))


def process(job: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    path, dataset = job
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    if dataset in CROP_DATASETS:
        rgb = crop_to_viewport(rgb)
    w, h = rgb.size
    top = rgb.crop((0, 0, w, max(1, int(h * TOP_FRACTION))))
    return letterbox(rgb), letterbox(top)


def main() -> None:
    df = pd.read_csv("data/processed/index.csv", keep_default_na=False)
    n = len(df)
    out = {v: np.lib.format.open_memmap(f"data/processed/img224_{v}.npy", mode="w+",
                                        dtype=np.uint8, shape=(n, SIZE, SIZE, 3)) for v in ("full", "top")}
    with ProcessPoolExecutor() as pool:
        jobs = zip(df.path, df.dataset)
        for i, (full, top) in enumerate(tqdm(pool.map(process, jobs, chunksize=16), total=n, mininterval=10)):
            out["full"][i], out["top"][i] = full, top
    for arr in out.values():
        arr.flush()
    print(f"Wrote data/processed/img224_full.npy and img224_top.npy, shape {(n, SIZE, SIZE, 3)}")


if __name__ == "__main__":
    main()
