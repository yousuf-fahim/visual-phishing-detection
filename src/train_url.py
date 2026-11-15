"""Phase 2: URL-feature baseline. Logistic regression, LightGBM and a small MLP.

Run: .venv/Scripts/python -m src.train_url
     (needs data/processed/url_features.csv from src.url_features)

Uses the grouped splits from Phase 1 and evaluates on VALIDATION ONLY (the test
set is opened once, in Phase 5). LightGBM and the MLP are run with 3 seeds
(42, 43, 44) and reported as mean +- std. Logistic regression is deterministic.

Outputs
  results/phase2/metrics_runs.csv        one row per (model, feature set, seed)
  results/phase2/metrics_summary.csv     mean and std per (model, feature set)
  results/phase2/feature_importance.csv/.png   LightGBM gain importance (full features, seed 42)
  results/phase2/val_predictions.csv     per-sample validation scores (full features, seed 42)
  models/url_lr.joblib, models/url_lgbm.txt, models/url_mlp.pt
"""
import random
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from src.metrics import binary_metrics
from src.url_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SEEDS = [42, 43, 44]
OUT = Path("results/phase2")
MODELS = Path("models")

# Feature sets for the shortcut ablation. Each removes features we suspect are
# artifacts of how Phish360 was collected rather than real phishing signals.
HOMEPAGE_STRUCTURE = ["is_root_path", "path_depth", "path_len", "query_len", "n_query_params",
                      "n_slashes", "url_len", "path_has_file_ext", "fragment_len"]
ABLATIONS = {
    "full": [],
    "no_https": ["is_https"],
    "no_tld": ["tld", "tld_len"],
    "no_www": ["has_www"],
    "no_homepage_structure": HOMEPAGE_STRUCTURE,
    "no_suspected_artifacts": ["has_www", "tld", "tld_len"] + HOMEPAGE_STRUCTURE,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    feats = pd.read_csv("data/processed/url_features.csv", keep_default_na=False)
    splits = pd.read_csv("data/processed/splits.csv", keep_default_na=False)
    d = feats.merge(splits, on="path")
    d = d[(d.dataset == "phish360") & (d.exclude == "")]
    return d[d.split == "train"].reset_index(drop=True), d[d.split == "val"].reset_index(drop=True)


def make_preprocessor(num: list, cat: list) -> ColumnTransformer:
    """Scale numbers; one-hot the TLD, folding TLDs seen < 10 times into 'infrequent'."""
    parts = [("num", StandardScaler(), num)]
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                                           sparse_output=False), cat))
    return ColumnTransformer(parts)


# ---------- the three models ----------

def fit_lr(train, val, num, cat, seed):
    model = make_pipeline(make_preprocessor(num, cat), LogisticRegression(max_iter=2000, C=1.0))
    model.fit(train[num + cat], train.label_binary)
    return model, model.predict_proba(val[num + cat])[:, 1]


def fit_lgbm(train, val, num, cat, seed):
    def prep(df):
        X = df[num + cat].copy()
        for c in cat:  # LightGBM handles categories natively; unseen TLDs become NaN
            X[c] = pd.Categorical(X[c], categories=sorted(train[c].unique()))
        return X
    model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               random_state=seed, verbose=-1)
    Xtr, Xva = prep(train), prep(val)
    model.fit(Xtr, train.label_binary, eval_set=[(Xva, val.label_binary)],
              callbacks=[lgb.early_stopping(100, verbose=False)])  # early stop on val log-loss
    return model, model.predict_proba(Xva)[:, 1]


class MLP(nn.Module):
    """Two hidden layers of 64 units. Small on purpose: about 30 input features."""

    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit_mlp(train, val, num, cat, seed, max_epochs=200, patience=15):
    set_seed(seed)
    pre = make_preprocessor(num, cat).fit(train[num + cat])
    Xtr = torch.tensor(pre.transform(train[num + cat]), dtype=torch.float32)
    Xva = torch.tensor(pre.transform(val[num + cat]), dtype=torch.float32)
    ytr = torch.tensor(train.label_binary.to_numpy(), dtype=torch.float32)
    yva = torch.tensor(val.label_binary.to_numpy(), dtype=torch.float32)

    model = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for batch in torch.randperm(len(Xtr), generator=gen).split(256):
            opt.zero_grad()
            loss_fn(model(Xtr[batch]), ytr[batch]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva).item()
        if val_loss < best - 1e-4:  # early stopping on validation loss
            best, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        score = torch.sigmoid(model(Xva)).numpy()
    return (model, pre), score


FITTERS = {"logistic_regression": fit_lr, "lightgbm": fit_lgbm, "mlp": fit_mlp}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    train, val = load_data()
    print(f"train={len(train)} val={len(val)} (phish share {train.label_binary.mean():.3f} / {val.label_binary.mean():.3f})")

    rows, saved = [], {}
    for variant, drop in ABLATIONS.items():
        num = [f for f in NUMERIC_FEATURES if f not in drop]
        cat = [f for f in CATEGORICAL_FEATURES if f not in drop]
        for model_name, fit in FITTERS.items():
            # LR is deterministic; only the full LR is needed once per variant.
            seeds = [42] if model_name == "logistic_regression" else SEEDS
            if variant != "full" and model_name == "mlp":
                continue  # ablation uses LR + LightGBM; MLP only on the full set
            for seed in seeds:
                set_seed(seed)
                model, score = fit(train, val, num, cat, seed)
                m = binary_metrics(val.label_binary, score)
                rows.append({"model": model_name, "features": variant, "seed": seed, **m})
                print(f"{variant:24s} {model_name:20s} seed={seed}  "
                      f"AUC={m['roc_auc']:.4f} F1={m['f1']:.4f} TPR@1%FPR={m['tpr_at_1pct_fpr']:.4f}")
                if variant == "full" and seed == 42:
                    saved[model_name] = (model, score, num, cat)

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "metrics_runs.csv", index=False)
    metric_cols = ["precision", "recall", "f1", "roc_auc", "tpr_at_1pct_fpr"]
    summary = runs.groupby(["features", "model"])[metric_cols].agg(["mean", "std"]).round(4)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary["n_seeds"] = runs.groupby(["features", "model"]).size().to_numpy()
    summary.to_csv(OUT / "metrics_summary.csv", index=False)
    print(summary.to_string())

    # Save the full-feature models (seed 42)
    joblib.dump(saved["logistic_regression"][0], MODELS / "url_lr.joblib")
    saved["lightgbm"][0].booster_.save_model(str(MODELS / "url_lgbm.txt"))
    mlp, pre = saved["mlp"][0]
    torch.save({"state_dict": mlp.state_dict(), "n_in": mlp.net[0].in_features}, MODELS / "url_mlp.pt")
    joblib.dump(pre, MODELS / "url_mlp_preprocessor.joblib")

    # Per-sample validation scores for later comparisons
    pd.DataFrame({"path": val.path, "label_binary": val.label_binary,
                  **{f"score_{k}": v[1] for k, v in saved.items()}}).to_csv(OUT / "val_predictions.csv", index=False)

    # LightGBM gain importance: how much each feature reduced the loss in total
    booster = saved["lightgbm"][0].booster_
    imp = pd.DataFrame({"feature": booster.feature_name(),
                        "gain": booster.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp["gain_share"] = (imp.gain / imp.gain.sum()).round(4)
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.feature, top.gain_share, color="#4C72B0")
    ax.set_xlabel("share of total gain")
    ax.set_title("URL LightGBM: top 15 features (Phish360 train, seed 42)")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=120)


if __name__ == "__main__":
    main()
