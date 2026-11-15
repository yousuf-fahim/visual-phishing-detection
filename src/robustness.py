"""Phase 7: how fragile is the screenshot model to everyday image changes?

Run: .venv/Scripts/python -m src.robustness

Each perturbation is applied to the Phish360 TEST screenshots (the 224x224 letterboxed
images; only the page content area is changed, never the grey padding) at 5 severity
levels, then scored by the fine-tuned MobileNetV3 (seeds 42-44). Late fusion re-averages
with the unchanged URL score, to show how much the URL half protects it.

  blur        Gaussian blur, sigma 0.5, 1, 2, 3, 4 pixels
  jpeg        JPEG re-compression at quality 90, 70, 50, 30, 10
  brightness  brightness x1.2, 1.4, 1.6, 1.8, 2.0 (washes the page out)
  grayscale   blend toward greyscale: 20, 40, 60, 80, 100%
  invert      colour inversion of 20, 40, 60, 80, 100% of the page area (top-down band)
  top_crop    keep only the top 80, 60, 40, 30, 20% of the page, grey below ("above the fold")

Severity 0 is the clean image. Metrics: accuracy at threshold 0.5, ROC-AUC, TPR@1%FPR.
Outputs: results/phase7/robustness.csv, robustness.png
"""
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter

from src.embed_images import to_input
from src.evaluate import SEEDS
from src.explain import CONTENT_ROWS, test_rows
from src.metrics import binary_metrics
from src.preprocess_images import PAD_RGB
from src.train_fusion import load_image_model

OUT = Path("results/phase7")
LEVELS = {
    "blur": [0.5, 1, 2, 3, 4],
    "jpeg": [90, 70, 50, 30, 10],
    "brightness": [1.2, 1.4, 1.6, 1.8, 2.0],
    "grayscale": [0.2, 0.4, 0.6, 0.8, 1.0],
    "invert": [0.2, 0.4, 0.6, 0.8, 1.0],
    "top_crop": [0.8, 0.6, 0.4, 0.3, 0.2],
}


def perturb(img: np.ndarray, kind: str, level: float) -> np.ndarray:
    """Apply one perturbation to the page content of one 224x224x3 uint8 image."""
    out = img.copy()
    page = Image.fromarray(np.ascontiguousarray(img[CONTENT_ROWS]))
    if kind == "blur":
        page = page.filter(ImageFilter.GaussianBlur(level))
    elif kind == "jpeg":
        buf = io.BytesIO()
        page.save(buf, format="JPEG", quality=int(level))
        page = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    elif kind == "brightness":
        page = ImageEnhance.Brightness(page).enhance(level)
    elif kind == "grayscale":
        page = Image.blend(page, page.convert("L").convert("RGB"), level)
    elif kind == "invert":
        a = np.asarray(page).copy()
        rows = int(round(level * a.shape[0]))
        a[:rows] = 255 - a[:rows]
        page = Image.fromarray(a)
    elif kind == "top_crop":
        a = np.asarray(page).copy()
        a[int(round(level * a.shape[0])):] = PAD_RGB
        page = Image.fromarray(a)
    out[CONTENT_ROWS] = np.asarray(page)
    return out


@torch.no_grad()
def score(model, imgs: np.ndarray) -> np.ndarray:
    return np.concatenate([torch.sigmoid(model.net(to_input(imgs[i:i + 128]))).squeeze(1).numpy()
                           for i in range(0, len(imgs), 128)])


def main() -> None:
    torch.set_num_threads(6)
    s, te, _ = test_rows()
    y = s.label_binary.to_numpy()[te]
    clean = np.array(np.load("data/processed/img224_full.npy", mmap_mode="r")[te])
    url = pd.read_csv("results/phase5/test_predictions.csv")["score_url_only:full"].to_numpy()  # seed-mean
    models = [load_image_model(sd) for sd in SEEDS]

    rows = []
    jobs = [("none", 0, 0.0)] + [(k, i + 1, lv) for k, lvs in LEVELS.items() for i, lv in enumerate(lvs)]
    for kind, sev, lv in jobs:
        imgs = clean if kind == "none" else np.stack([perturb(im, kind, lv) for im in clean])
        for sd, m in zip(SEEDS, models):
            p = score(m, imgs)
            for name, sc in (("image_only", p), ("late_fusion", (p + url) / 2)):
                met = binary_metrics(y, sc)
                rows.append({"perturbation": kind, "severity": sev, "level": lv, "model": name, "seed": sd,
                             "accuracy": float(((sc >= 0.5) == y).mean()), **met})
        print(f"{kind:10s} severity {sev} done", flush=True)

    df = pd.DataFrame(rows)
    # severity 0 (clean) is shared by every perturbation curve
    clean_rows = df[df.perturbation == "none"]
    df = pd.concat([df[df.perturbation != "none"]] +
                   [clean_rows.assign(perturbation=k) for k in LEVELS])
    summ = df.groupby(["perturbation", "severity", "level", "model"])[["accuracy", "roc_auc", "tpr_at_1pct_fpr"]] \
             .agg(["mean", "std"]).round(4)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(OUT / "robustness.csv", index=False)
    print(summ[summ.model == "image_only"].to_string(index=False))

    fig, axes = plt.subplots(2, len(LEVELS), figsize=(4 * len(LEVELS), 7), sharey="row")
    for j, kind in enumerate(LEVELS):
        sub = summ[summ.perturbation == kind]
        for i, metric in enumerate(["accuracy", "tpr_at_1pct_fpr"]):
            ax = axes[i, j]
            for name, color in (("image_only", "#DD8452"), ("late_fusion", "#8172B3")):
                t = sub[sub.model == name].sort_values("severity")
                ax.errorbar(t.severity, t[f"{metric}_mean"], yerr=t[f"{metric}_std"], marker="o",
                            color=color, label=name, capsize=3)
            ax.set_xticks(range(6), ["clean"] + [str(v) for v in LEVELS[kind]], fontsize=7)
            if i == 0:
                ax.set_title(kind)
            if j == 0:
                ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Phish360 test: performance vs perturbation severity (mean ± std over 3 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "robustness.png", dpi=110)


if __name__ == "__main__":
    main()
