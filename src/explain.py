"""Phase 7: what does the screenshot model look at?

Run: .venv/Scripts/python -m src.explain

Model: fine-tuned MobileNetV3-small (full page, seed 42 for pictures; seeds 42-44 for numbers).
Data: Phish360 TEST split (already final, so nothing here changes any model choice).

1. Grad-CAM. For one image, take the gradient of the model's output for the predicted class
   with respect to the last convolutional feature maps. Feature maps whose activation
   increases the score get high weight; the weighted sum is a coarse heatmap of
   "where the evidence was". We show 20 examples (5 each of TP, TN, FP, FN, most confident)
   and the average heatmap over all test pages, per true class.
2. Occlusion. Cover one 32x32 square with neutral grey, re-score, and record how much the
   phishing probability moves. Repeat for every square (a 7x7 grid) and every test page.
   The average |change| per square shows which regions matter most.
3. Border vs centre. New frozen-ResNet18 + logistic-regression classifiers that see only the
   outer 20-pixel ring of the page, or only the central box, compared with the full page
   and the trivial-feature floor. If the border alone works well, the model could be
   reading browser chrome or page edges rather than content.

Outputs (results/phase7/)
  gradcam_mean.png, occlusion_mean.png, border_center.csv, region_summary.csv (committable: aggregates)
  screenshots/gradcam_examples.png, screenshots/gradcam_examples.csv (git-ignored: real screenshots)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.embed_images import to_input
from src.evaluate import evaluate_with_ci
from src.preprocess_images import PAD_RGB
from src.train_fusion import load_image_model

OUT = Path("results/phase7")
SHOTS = OUT / "screenshots"
CONTENT_ROWS = slice(28, 196)  # a 1280x960 page letterboxed to 224x168, centred vertically
PATCH = 32


def test_rows():
    s = pd.read_csv("data/processed/splits.csv", keep_default_na=False)
    m = (s.dataset == "phish360") & (s.exclude == "")
    return s, np.flatnonzero(m & (s.split == "test")), np.flatnonzero(m & (s.split == "train"))


# ---------------- 1. Grad-CAM ----------------

def gradcam_maps(model, images, rows) -> np.ndarray:
    """Grad-CAM for the predicted class of each image. Returns (n, 224, 224) in [0, 1]."""
    cam = GradCAM(model=model.net, target_layers=[model.net.blocks[-1]])
    maps = []
    for start in range(0, len(rows), 32):
        r = rows[start:start + 32]
        x = to_input(images[r])
        with torch.no_grad():
            pred = (torch.sigmoid(model.net(x)).squeeze(1) >= 0.5).long().tolist()
        maps.append(cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(p) for p in pred]))
    return np.concatenate(maps)


def region_shares(maps: np.ndarray) -> pd.DataFrame:
    """Share of heatmap mass in named regions of the page content (rows 28-196)."""
    m = maps[:, CONTENT_ROWS, :]
    total = m.sum((1, 2)) + 1e-9
    h, w = m.shape[1:]
    regions = {
        "top 20% (header/logo)": m[:, : int(0.2 * h), :],
        "middle 60%": m[:, int(0.2 * h): int(0.8 * h), :],
        "bottom 20% (footer)": m[:, int(0.8 * h):, :],
        "left/right 15% edges": np.concatenate([m[:, :, : int(0.15 * w)], m[:, :, int(0.85 * w):]], axis=2),
        "central 50% box": m[:, int(0.25 * h): int(0.75 * h), int(0.25 * w): int(0.75 * w)],
    }
    return pd.DataFrame({k: v.sum((1, 2)) / total for k, v in regions.items()})


# ---------------- 2. Occlusion ----------------

@torch.no_grad()
def occlusion_maps(model, images, rows) -> np.ndarray:
    """(n, 7, 7) mean |change in phishing probability| when each 32x32 square is greyed out."""
    n_grid = 224 // PATCH
    out = np.zeros((len(rows), n_grid, n_grid), dtype=np.float32)
    for start in range(0, len(rows), 16):
        r = rows[start:start + 16]
        base_imgs = np.array(images[r])
        base = torch.sigmoid(model.net(to_input(base_imgs))).squeeze(1).numpy()
        for gi in range(n_grid):
            for gj in range(n_grid):
                occ = base_imgs.copy()
                occ[:, gi * PATCH:(gi + 1) * PATCH, gj * PATCH:(gj + 1) * PATCH] = PAD_RGB
                p = torch.sigmoid(model.net(to_input(occ))).squeeze(1).numpy()
                out[start:start + len(r), gi, gj] = np.abs(base - p)
    return out


# ---------------- 3. Border vs centre ----------------

def masked(imgs: np.ndarray, keep: str) -> np.ndarray:
    """Keep only the 20-px ring of the page content ('border') or its central box ('center')."""
    out = np.empty_like(imgs)
    out[:] = PAD_RGB
    top, bottom = CONTENT_ROWS.start, CONTENT_ROWS.stop
    if keep == "border":
        b = 20
        out[:, top:bottom] = imgs[:, top:bottom]
        out[:, top + b:bottom - b, b:224 - b] = PAD_RGB
    else:  # centre: middle half of the content height and width
        h, w = bottom - top, 224
        r0, r1, c0, c1 = top + h // 4, bottom - h // 4, w // 4, w - w // 4
        out[:, r0:r1, c0:c1] = imgs[:, r0:r1, c0:c1]
    return out


@torch.no_grad()
def resnet_embed(imgs_fn, rows) -> np.ndarray:
    net = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=0).eval()
    return np.concatenate([net(to_input(imgs_fn(rows[i:i + 64]))).numpy() for i in range(0, len(rows), 64)])


def main() -> None:
    torch.set_num_threads(6)
    SHOTS.mkdir(parents=True, exist_ok=True)
    s, te, tr = test_rows()
    y = s.label_binary.to_numpy()
    images = np.load("data/processed/img224_full.npy", mmap_mode="r")
    model = load_image_model(42)
    preds = pd.read_csv("results/phase5/test_predictions.csv")
    assert (preds.path.to_numpy() == s.path.to_numpy()[te]).all()

    # ---- 1. Grad-CAM over all test pages + examples ----
    maps = gradcam_maps(model, images, te)
    shares = region_shares(maps)
    shares["true_label"] = np.where(y[te] == 1, "phishing", "legitimate")
    summary = shares.groupby("true_label").mean().round(3)
    # what an evenly spread heatmap would give, for reference
    uniform = region_shares(np.ones((1, 224, 224))).iloc[0].round(3).rename("uniform (reference)")
    summary = pd.concat([summary, uniform.to_frame().T])
    summary.to_csv(OUT / "region_summary.csv")
    print("Grad-CAM mass by region:\n", summary.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, lab, name in zip(axes, (0, 1), ("legitimate", "phishing")):
        im = ax.imshow(maps[y[te] == lab].mean(0)[CONTENT_ROWS], cmap="jet", vmin=0, vmax=0.6)
        ax.set_title(f"mean Grad-CAM, true {name} (n={(y[te] == lab).sum()})")
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.03)
    fig.savefig(OUT / "gradcam_mean.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    score = preds["score_image_only"].to_numpy()  # seed-averaged test scores from Phase 5
    yt = y[te]
    kinds = {"TP": (yt == 1) & (score >= 0.5), "TN": (yt == 0) & (score < 0.5),
             "FP": (yt == 0) & (score >= 0.5), "FN": (yt == 1) & (score < 0.5)}
    picks = []
    for k, m in kinds.items():  # most confident examples of each kind
        idx = np.flatnonzero(m)
        conf = np.abs(score[idx] - 0.5)
        picks += [(k, i) for i in idx[np.argsort(-conf)[:5]]]
    fig, axes = plt.subplots(4, 5, figsize=(20, 13))
    rows_out = []
    for ax, (k, i) in zip(axes.ravel(), picks):
        ax.imshow(images[te[i]][CONTENT_ROWS])
        ax.imshow(maps[i][CONTENT_ROWS], cmap="jet", alpha=0.45)
        ax.set_title(f"{k}  p(phish)={score[i]:.2f}", fontsize=10)
        ax.axis("off")
        rows_out.append({"kind": k, "test_index": int(i), "path": preds.path[i], "score_image": round(score[i], 3),
                         "score_url": round(preds["score_url_only:full"][i], 3), **shares.iloc[i].drop("true_label").round(3)})
    fig.tight_layout()
    fig.savefig(SHOTS / "gradcam_examples.png", dpi=90)
    plt.close(fig)
    pd.DataFrame(rows_out).to_csv(SHOTS / "gradcam_examples.csv", index=False)

    # ---- 2. Occlusion over all test pages ----
    occ_path = OUT / "occlusion_maps_s42.npy"
    if occ_path.exists():  # slow step (~78k forward passes): reuse a previous run's maps
        occ = np.load(occ_path)
    else:
        occ = occlusion_maps(model, images, te)
        np.save(occ_path, occ)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    vmax = occ.mean(0).max()
    for ax, lab, name in zip(axes, (0, 1), ("legitimate", "phishing")):
        im = ax.imshow(occ[yt == lab].mean(0), cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"true {name}: mean |change in p(phish)|\nwhen one 32x32 square is greyed out", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.03)
    fig.savefig(OUT / "occlusion_mean.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    grid = occ.mean(0)
    print("Occlusion: mean |change in p| per 32px row band (top->bottom):", grid.mean(1).round(4).tolist())
    pd.DataFrame(grid, index=[f"row{i}" for i in range(grid.shape[0])],
                 columns=[f"col{j}" for j in range(grid.shape[1])]).round(5).to_csv(OUT / "occlusion_grid.csv")

    # ---- 3. Border vs centre (frozen ResNet18 + logistic regression, trained on train) ----
    rows = []
    full = np.load("data/processed/emb_resnet18_full.npy")
    variants = {"full page": (full[tr], full[te])}
    for keep in ("border", "center"):
        fn = lambda r, k=keep: masked(np.array(images[r]), k)
        variants[f"{keep} only"] = (resnet_embed(fn, tr), resnet_embed(fn, te))
    for name, (Xtr, Xte) in variants.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1][None, :]
        rows.append({"input": name, "model": "frozen ResNet18 + logistic regression", **evaluate_with_ci(yt, p)})
    floor = pd.read_csv("results/final_in_distribution.csv").query("model == 'trivial_floor'").iloc[0]
    rows.append({"input": "trivial statistics (Phase 1 features)", "model": "LightGBM (from Phase 5)",
                 **{k: floor[k] for k in floor.index if k.startswith(("precision", "recall", "f1", "roc_auc", "tpr"))}})
    bc = pd.DataFrame(rows)
    bc.to_csv(OUT / "border_center.csv", index=False)
    print(bc[["input", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "tpr_at_1pct_fpr"]].to_string(index=False))


if __name__ == "__main__":
    main()
