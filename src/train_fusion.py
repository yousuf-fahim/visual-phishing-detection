"""Phase 4: fuse screenshot and URL information. Ablation and robustness probe.

Run: .venv/Scripts/python -m src.train_fusion   (after Phases 2 and 3)

Models compared on the VALIDATION set, seeds 42/43/44 (seed s uses image model seed s):
  url_only      LightGBM on URL features (Phase 2 recipe)
  image_only    fine-tuned MobileNetV3-small on the full screenshot (Phase 3 best)
  early_fusion  one MLP on [image embedding, URL features] concatenated
  late_fusion   average of the URL and image probabilities (no weights tuned, so nothing is fitted on val)

Each is run with two URL feature sets: 'full' and 'no_suspected_artifacts' (docs/QUESTIONS.md Q8),
and with two image embeddings for early fusion: the fine-tuned MobileNetV3 (main) and frozen
ImageNet ResNet18 (a check: a fine-tuned model has partly memorised its training images, so its
training-set embeddings look cleaner than on new pages, and a fusion head may over-trust them).

Robustness probe (at inference only, models unchanged):
  url_missing   URL features replaced by their training mean (= zeros after standardisation)
  image_blank   screenshot replaced by a blank white page on the same 224x224 canvas

Outputs: results/phase4/{ablation_runs,ablation_summary,robustness_runs,robustness_summary}.csv,
         results/phase4/ablation.png, data/processed/emb_ftmobilenetv3_full_s{seed}.npy
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.metrics import binary_metrics
from src.preprocess_images import PAD_RGB
from src.train_image import EVAL_TF, TimmBinary
from src.train_url import ABLATIONS, fit_lgbm, make_preprocessor, set_seed
from src.url_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SEEDS = [42, 43, 44]
OUT = Path("results/phase4")
METRICS = ["precision", "recall", "f1", "roc_auc", "tpr_at_1pct_fpr"]


# ---------- image side ----------

def load_image_model(seed: int, run_prefix: str = "") -> TimmBinary:
    """Load a fine-tuned MobileNetV3 checkpoint, e.g. models/img_mobilenetv3_full_s42.pt,
    or with run_prefix='phish-iris-all-' models/img_phish-iris-all-mobilenetv3_full_s42.pt."""
    # Same architecture as in training; our fine-tuned weights then overwrite the ImageNet ones.
    model = TimmBinary("mobilenetv3_small_100.lamb_in1k")
    path = f"models/img_{run_prefix}mobilenetv3_full_s{seed}.pt"
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval()


@torch.no_grad()
def image_outputs(model: TimmBinary, imgs_uint8) -> tuple[np.ndarray, np.ndarray]:
    """Return (embedding before the last layer, phishing probability) for a stack of images."""
    embs, probs = [], []
    for start in range(0, len(imgs_uint8), 128):
        x = torch.from_numpy(np.ascontiguousarray(imgs_uint8[start:start + 128])).permute(0, 3, 1, 2)
        x = EVAL_TF(x)
        feats = model.net.forward_head(model.net.forward_features(x), pre_logits=True)
        embs.append(feats.numpy())
        probs.append(torch.sigmoid(model.net.get_classifier()(feats).squeeze(1)).numpy())
    return np.concatenate(embs), np.concatenate(probs)


def blank_canvas() -> np.ndarray:
    """A white 1280x960 page letterboxed exactly like real screenshots (224x168 white, grey pad)."""
    canvas = np.empty((1, 224, 224, 3), dtype=np.uint8)
    canvas[:] = PAD_RGB
    canvas[0, 28:196] = 255  # 168 rows of content centred vertically
    return canvas


# ---------- early-fusion MLP ----------

class FusionMLP(nn.Module):
    """Input: [image embedding, URL features] concatenated into one vector."""

    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(n_in, 256), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit_early(Xtr, ytr, Xva, yva, seed, max_epochs=200, patience=15) -> nn.Module:
    set_seed(seed)
    Xtr_t, Xva_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
    ytr_t, yva_t = torch.tensor(ytr, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    model = FusionMLP(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for b in torch.randperm(len(Xtr_t), generator=gen).split(128):
            opt.zero_grad()
            loss_fn(model(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva_t), yva_t).item()
        if vl < best - 1e-4:  # early stopping on validation loss
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def predict_early(model, X) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


# ---------- main ----------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    splits = pd.read_csv("data/processed/splits.csv", keep_default_na=False)
    urlf = pd.read_csv("data/processed/url_features.csv", keep_default_na=False)
    assert (splits.path == urlf.path).all()
    d = pd.concat([urlf, splits.drop(columns="path")], axis=1)
    use = (d.dataset == "phish360") & (d.exclude == "")
    tr = np.flatnonzero(use & (d.split == "train"))
    va = np.flatnonzero(use & (d.split == "val"))
    y = d.label_binary.to_numpy()
    train, val = d.iloc[tr].reset_index(drop=True), d.iloc[va].reset_index(drop=True)
    images = np.load("data/processed/img224_full.npy", mmap_mode="r")
    frozen_resnet = np.load("data/processed/emb_resnet18_full.npy")

    rows, robust = [], []
    for seed in SEEDS:
        # Image model for this seed: embeddings for every sample (saved for later phases) + blank-page output
        emb_path = Path(f"data/processed/emb_ftmobilenetv3_full_s{seed}.npy")
        prob_path = Path(f"data/processed/prob_ftmobilenetv3_full_s{seed}.npy")
        img_model = load_image_model(seed)
        if emb_path.exists() and prob_path.exists():
            emb_all, prob_all = np.load(emb_path), np.load(prob_path)
        else:
            emb_all, prob_all = image_outputs(img_model, images)
            np.save(emb_path, emb_all)
            np.save(prob_path, prob_all)
        blank_emb, blank_prob = image_outputs(img_model, blank_canvas())
        img_va = prob_all[va]
        rows.append({"model": "image_only", "url_features": "-", "image_embedding": "ft-mobilenetv3",
                     "seed": seed, **binary_metrics(y[va], img_va)})
        robust.append({"model": "image_only", "url_features": "-", "image_embedding": "ft-mobilenetv3",
                       "seed": seed, "probe": "image_blank",
                       **binary_metrics(y[va], np.full(len(va), blank_prob[0]))})

        for feat_set, drop in [("full", ABLATIONS["full"]), ("no_suspected_artifacts", ABLATIONS["no_suspected_artifacts"])]:
            num = [f for f in NUMERIC_FEATURES if f not in drop]
            cat = [f for f in CATEGORICAL_FEATURES if f not in drop]

            # URL-only LightGBM (also the URL half of late fusion)
            set_seed(seed)
            lgbm, url_va = fit_lgbm(train, val, num, cat, seed)
            rows.append({"model": "url_only", "url_features": feat_set, "image_embedding": "-",
                         "seed": seed, **binary_metrics(y[va], url_va)})
            # URL "missing": every feature set to its training mean (TLD set to the most common one)
            val_missing = val.copy()
            for c in num:
                val_missing[c] = train[c].mean()
            for c in cat:
                val_missing[c] = train[c].mode()[0]
            Xmiss = val_missing[num + cat].copy()
            for c in cat:
                Xmiss[c] = pd.Categorical(Xmiss[c], categories=sorted(train[c].unique()))
            url_va_missing = lgbm.predict_proba(Xmiss)[:, 1]
            robust.append({"model": "url_only", "url_features": feat_set, "image_embedding": "-", "seed": seed,
                           "probe": "url_missing", **binary_metrics(y[va], url_va_missing)})

            # Late fusion: plain average of the two probabilities
            for probe, u, im in [("none", url_va, img_va), ("url_missing", url_va_missing, img_va),
                                 ("image_blank", url_va, np.full(len(va), blank_prob[0]))]:
                rec = {"model": "late_fusion", "url_features": feat_set, "image_embedding": "ft-mobilenetv3",
                       "seed": seed, **binary_metrics(y[va], (u + im) / 2)}
                (rows if probe == "none" else robust).append(rec if probe == "none" else {**rec, "probe": probe})

            # Early fusion: standardised URL features + standardised image embedding -> MLP
            pre = make_preprocessor(num, cat).fit(train[num + cat])
            U_tr, U_va = pre.transform(train[num + cat]), pre.transform(val[num + cat])
            for emb_name, E in [("ft-mobilenetv3", emb_all), ("frozen-resnet18", frozen_resnet)]:
                mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-6
                I_tr, I_va = (E[tr] - mu) / sd, (E[va] - mu) / sd
                model = fit_early(np.hstack([I_tr, U_tr]), y[tr], np.hstack([I_va, U_va]), y[va], seed)
                rows.append({"model": "early_fusion", "url_features": feat_set, "image_embedding": emb_name,
                             "seed": seed, **binary_metrics(y[va], predict_early(model, np.hstack([I_va, U_va])))})
                if emb_name == "ft-mobilenetv3":
                    if feat_set == "full" and seed == 42:
                        torch.save(model.state_dict(), "models/fusion_early_s42.pt")
                    # probes: URL block -> 0 (= training mean after scaling); image -> blank page embedding
                    blank_I = np.repeat((blank_emb - mu) / sd, len(va), axis=0)
                    for probe, X in [("url_missing", np.hstack([I_va, np.zeros_like(U_va)])),
                                     ("image_blank", np.hstack([blank_I, U_va]))]:
                        robust.append({"model": "early_fusion", "url_features": feat_set,
                                       "image_embedding": emb_name, "seed": seed, "probe": probe,
                                       **binary_metrics(y[va], predict_early(model, X))})
            print(f"seed {seed} {feat_set}: done", flush=True)

    def summarise(df, keys):
        s = df.groupby(keys)[METRICS].agg(["mean", "std"]).round(4)
        s.columns = [f"{m}_{k}" for m, k in s.columns]
        s["n_seeds"] = df.groupby(keys).size()
        return s.reset_index()

    runs, rob = pd.DataFrame(rows), pd.DataFrame(robust)
    runs.to_csv(OUT / "ablation_runs.csv", index=False)
    rob.to_csv(OUT / "robustness_runs.csv", index=False)
    summ = summarise(runs, ["model", "url_features", "image_embedding"])
    summ.to_csv(OUT / "ablation_summary.csv", index=False)
    # robustness: put the clean score next to each probe
    rsum = summarise(rob, ["model", "url_features", "image_embedding", "probe"])
    clean = summ[["model", "url_features", "image_embedding", "roc_auc_mean", "tpr_at_1pct_fpr_mean"]].rename(
        columns={"roc_auc_mean": "clean_roc_auc", "tpr_at_1pct_fpr_mean": "clean_tpr_at_1pct_fpr"})
    rsum = rsum.merge(clean, on=["model", "url_features", "image_embedding"], how="left")
    rsum.to_csv(OUT / "robustness_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ[["model", "url_features", "image_embedding", "roc_auc_mean", "roc_auc_std",
                "tpr_at_1pct_fpr_mean", "tpr_at_1pct_fpr_std", "f1_mean"]].to_string(index=False))
    print(rsum[["model", "url_features", "probe", "clean_roc_auc", "roc_auc_mean",
                "clean_tpr_at_1pct_fpr", "tpr_at_1pct_fpr_mean"]].to_string(index=False))

    # Ablation chart (main configuration: full URL features, fine-tuned image model)
    main_rows = summ[(summ.url_features.isin(["full", "-"])) & (summ.image_embedding.isin(["ft-mobilenetv3", "-"]))]
    order = ["url_only", "image_only", "early_fusion", "late_fusion"]
    main_rows = main_rows.set_index("model").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, m in zip(axes, ["roc_auc", "tpr_at_1pct_fpr"]):
        ax.bar(order, main_rows[f"{m}_mean"], yerr=main_rows[f"{m}_std"], color=["#4C72B0", "#DD8452", "#55A868", "#8172B3"])
        ax.set_title(f"validation {m} (mean ± std, 3 seeds)")
        ax.set_ylim(0.0 if m != "roc_auc" else 0.9, 1.0)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=120)


if __name__ == "__main__":
    main()
