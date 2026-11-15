"""Phase 1: one pass over every screenshot to compute
  (a) trivial "shortcut" features, used by src/shortcut_check.py
  (b) a perceptual hash (pHash), used by src/dedup.py

Run: .venv/Scripts/python -m src.image_stats
Output: data/processed/image_features.csv (one row per index.csv row, same order)

Perceptual hash in plain words: shrink the image to 32x32 greyscale, take its
low-frequency pattern (DCT), and keep 64 bits saying "brighter or darker than
average" for each part. Two screenshots of the same page give almost the same 64
bits even if they're re-compressed or slightly changed, so a small Hamming
distance (number of differing bits) means "near-duplicate".
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

INDEX = Path("data/processed/index.csv")
OUT = Path("data/processed/image_features.csv")


def features(path: str) -> dict:
    """Return trivial features + pHash for one screenshot (or NaNs if unreadable)."""
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            phash = str(imagehash.phash(rgb))  # 64-bit hash as 16 hex chars
            # Colour statistics on a 320-px-wide copy: same averages, ~16x faster.
            small = rgb.copy()
            small.thumbnail((320, 320))
            a = np.asarray(small, dtype=np.float32)
        gray = a.mean(axis=2)
        return {
            "phash": phash,
            "mean_r": a[..., 0].mean(), "mean_g": a[..., 1].mean(), "mean_b": a[..., 2].mean(),
            # near-white pixels: all three channels above 240 (of 255)
            "white_frac": float((a.min(axis=2) > 240).mean()),
            "gray_std": float(gray.std()),  # overall contrast: also a trivial statistic
        }
    except Exception:
        return {"phash": "", "mean_r": np.nan, "mean_g": np.nan, "mean_b": np.nan,
                "white_frac": np.nan, "gray_std": np.nan}


def main() -> None:
    df = pd.read_csv(INDEX, keep_default_na=False)
    with ProcessPoolExecutor() as pool:
        feats = list(tqdm(pool.map(features, df.path, chunksize=32), total=len(df)))
    out = pd.DataFrame(feats)
    out.insert(0, "path", df.path)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(out)} rows, unreadable={int((out.phash == '').sum())}")


if __name__ == "__main__":
    main()
