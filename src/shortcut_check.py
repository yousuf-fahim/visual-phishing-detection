"""Phase 1, step 4: can trivial image statistics predict the label?

Run: .venv/Scripts/python -m src.shortcut_check   (after image_stats and dedup)

Idea in plain words: if "file size" or "how white the page is" alone separates
phishing from legitimate, a CNN can score highly by learning those things instead
of real phishing cues. We measure how far trivial features alone get, using
grouped cross-validation (no near-duplicate or same-domain pair is split across
folds), so the result is a floor the real models must clearly beat.

Outputs
  results/phase1/shortcut_check.json         metrics per model + per single feature
  results/phase1/shortcut_features.png       distributions by label
  data/processed/shortcut_oof.csv            out-of-fold trivial-model score per sample,
                                             used later to build a "shortcut-hard" subset
"""
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path("results/phase1")
FEATURES = ["width", "height", "aspect", "file_kb", "mean_r", "mean_g", "mean_b", "white_frac", "gray_std",
            "is_jpeg"]


def tpr_at_fpr(y, score, max_fpr=0.01) -> float:
    """True-positive rate (share of phishing caught) when at most 1% of legit pages are flagged."""
    fpr, tpr, _ = roc_curve(y, score)
    return float(tpr[fpr <= max_fpr].max())


def metrics(y, score) -> dict:
    pred = (score >= 0.5).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, score), 4),
        "f1": round(f1_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "tpr_at_1pct_fpr": round(tpr_at_fpr(y, score), 4),
    }


def oof_scores(model_fn, X, y, groups) -> np.ndarray:
    """Out-of-fold predictions: each sample is scored by a model that never saw its group."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        m = model_fn().fit(X[tr], y[tr])
        scores[te] = m.predict_proba(X[te])[:, 1]
    return scores


def main() -> None:
    idx = pd.read_csv("data/processed/index.csv", keep_default_na=False)
    feats = pd.read_csv("data/processed/image_features.csv", keep_default_na=False)
    groups = pd.read_csv("data/processed/groups.csv")
    df = idx.merge(feats, on="path").merge(groups, on="path")
    df["aspect"] = df.width / df.height
    df["file_kb"] = df.file_bytes / 1024

    results, oof_frames = {}, []
    for name, d in df.groupby("dataset"):
        X = d[FEATURES].to_numpy(float)
        y = d.label_binary.to_numpy(int)
        g = d.group_id.to_numpy()
        res = {"n": len(d), "phishing_fraction": round(y.mean(), 4)}

        # Each feature on its own: AUC of the raw value (0.5 = useless, 1.0 or 0.0 = perfect).
        res["single_feature_auc"] = {f: round(roc_auc_score(y, d[f]), 4) for f in FEATURES}

        lr = oof_scores(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y, g)
        gbm = oof_scores(lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                                    random_state=SEED, verbose=-1), X, y, g)
        res["logistic_regression"] = metrics(y, lr)
        res["lightgbm"] = metrics(y, gbm)

        # Without the two content-density features (file size, whiteness) to see what's left.
        rest = [f for f in FEATURES if f not in ("file_kb", "white_frac")]
        gbm_rest = oof_scores(lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                                         random_state=SEED, verbose=-1),
                              d[rest].to_numpy(float), y, g)
        res["lightgbm_without_filesize_whiteness"] = metrics(y, gbm_rest)

        # Near-blank screenshots (almost all white or a flat colour): likely failed page loads.
        blank = (d.white_frac > 0.97) | (d.gray_std < 5)
        res["near_blank_screenshots"] = {("phish" if k else "legit"): int(v)
                                         for k, v in d[blank].label_binary.value_counts().items()}
        results[name] = res
        oof_frames.append(pd.DataFrame({"path": d.path, "shortcut_score": gbm}))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shortcut_check.json").write_text(json.dumps(results, indent=2))
    pd.concat(oof_frames).to_csv("data/processed/shortcut_oof.csv", index=False)
    print(json.dumps(results, indent=2))

    # Distributions of the most telling features, by label (Phish360 only, as the title says)
    p360 = df[df.dataset == "phish360"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, f in zip(axes, ["file_kb", "white_frac", "gray_std", "mean_g"]):
        for lab, color, txt in [(0, "#4C72B0", "legitimate"), (1, "#DD8452", "phishing")]:
            vals = p360.loc[p360.label_binary == lab, f]
            if f == "file_kb":
                vals = np.log10(vals.clip(lower=1))
            ax.hist(vals, bins=60, alpha=0.6, color=color, label=txt, density=True)
        ax.set_title("log10(file size, KB)" if f == "file_kb" else f)
    axes[0].legend()
    fig.suptitle("Phish360: trivial screenshot statistics by label")
    fig.tight_layout()
    fig.savefig(OUT / "shortcut_features.png", dpi=110)


if __name__ == "__main__":
    main()
