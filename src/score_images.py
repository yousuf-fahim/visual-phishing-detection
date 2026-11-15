"""Score a folder of screenshots with the same model and preprocessing as the web demo.

Run: .venv/Scripts/python -m src.score_images test-images
     (writes <folder>/scores.csv and prints a table)

Uses web/public/image_model.onnx (the exact file the demo ships) and the training letterbox
from src/preprocess_images.py, so the numbers match what the website shows for the same file.
Only for images you already have on disk; nothing is fetched.
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from src.preprocess_images import letterbox

MODEL = Path("web/public/image_model.onnx")


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "test-images")
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    rows = []
    for f in sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}):
        with Image.open(f) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
        x = letterbox(rgb).transpose(2, 0, 1)[None].astype(np.float32)
        p = float(sess.run(None, {"image": x})[0][0])
        rows.append({"file": f.name, "width": w, "height": h, "phishing_likeness": round(p, 4),
                     "model_says": "looks like phishing" if p >= 0.5 else "looks legitimate"})
    df = pd.DataFrame(rows)
    df.to_csv(folder / "scores.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
