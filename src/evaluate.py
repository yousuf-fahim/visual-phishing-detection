"""Phase 5: the final, one-time evaluation on the Phish360 TEST split.

Run: .venv/Scripts/python -m src.evaluate

Every model here was chosen on the validation set in Phases 1-4 (docs/QUESTIONS.md Q11).
Models are refitted exactly as before (same data, same seeds, deterministic), or loaded
from their Phase 3 checkpoints, then scored on the test split. Nothing is tuned here.

Models (seeds 42/43/44)
  trivial_floor           LightGBM on Phase 1 trivial image statistics (the shortcut floor)
  url_only                LightGBM on URL features            (full / no_suspected_artifacts)
  image_only              fine-tuned MobileNetV3, full page
  early_fusion_frozen     MLP on [frozen ResNet18 embedding, URL features] (full)
  late_fusion             mean of url_only and image_only probabilities (full / no_suspected_artifacts)

Uncertainty: 95% bootstrap confidence intervals. Resample the test set 1,000 times
with replacement; in each resample compute the metric for each seed and average.
The 2.5th and 97.5th percentiles of those 1,000 values form the interval.

Shortcut-hard subset: test pages the trivial-feature model gets WRONG (busy phishing
pages and sparse legitimate pages). If a model only learned page density, it fails here.

Outputs
  results/final_in_distribution.csv            all test pages
  results/phase5/final_shortcut_hard.csv       shortcut-hard subset
  results/phase5/test_predictions.csv          per-page seed-averaged scores (for Phase 7)
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.metrics import binary_metrics
from src.shortcut_check import FEATURES as TRIVIAL_FEATURES
from src.train_fusion import fit_early, image_outputs, load_image_model, predict_early
from src.train_url import ABLATIONS, fit_lgbm, make_preprocessor, set_seed
from src.url_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SEEDS = [42, 43, 44]
N_BOOT = 1000
METRICS = ["precision", "recall", "f1", "roc_auc", "tpr_at_1pct_fpr"]
OUT = Path("results/phase5")


def lgbm_predict(model, df, num, cat, train) -> np.ndarray:
    X = df[num + cat].copy()
    for c in cat:
        X[c] = pd.Categorical(X[c], categories=sorted(train[c].unique()))
    return model.predict_proba(X)[:, 1]


def image_probs_all_rows(seed: int, images, run_prefix: str = "") -> tuple[np.ndarray, np.ndarray]:
    """(embeddings, probabilities) of a fine-tuned image model for every row, cached on disk."""
    tag = f"{run_prefix}ftmobilenetv3_full_s{seed}"
    emb_p, prob_p = Path(f"data/processed/emb_{tag}.npy"), Path(f"data/processed/prob_{tag}.npy")
    if prob_p.exists() and len(np.load(prob_p, mmap_mode="r")) == len(images):
        return np.load(emb_p), np.load(prob_p)
    emb, prob = image_outputs(load_image_model(seed, run_prefix), images)
    np.save(emb_p, emb)
    np.save(prob_p, prob)
    return emb, prob


def evaluate_with_ci(y: np.ndarray, seed_scores: np.ndarray, rng_seed: int = 42) -> dict:
    """Point estimate = mean over seeds; CI = bootstrap over test pages (seed-averaged inside)."""
    point = {m: float(np.mean([binary_metrics(y, s)[m] for s in seed_scores])) for m in METRICS}
    rng = np.random.default_rng(rng_seed)
    boots = {m: [] for m in METRICS}
    n = len(y)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():  # need both classes for AUC
            continue
        per_seed = [binary_metrics(y[i], s[i]) for s in seed_scores]
        for m in METRICS:
            boots[m].append(np.mean([r[m] for r in per_seed]))
    out = {}
    for m in METRICS:
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        out[m], out[f"{m}_ci_low"], out[f"{m}_ci_high"] = round(point[m], 4), round(lo, 4), round(hi, 4)
    return out


def build_test_scores():
    """Return (y_test, test_paths, {model_name: array (n_seeds, n_test)})."""
    splits = pd.read_csv("data/processed/splits.csv", keep_default_na=False)
    urlf = pd.read_csv("data/processed/url_features.csv", keep_default_na=False)
    trivial = pd.read_csv("data/processed/image_features.csv", keep_default_na=False)
    idx = pd.read_csv("data/processed/index.csv", keep_default_na=False)
    assert (splits.path == urlf.path).all() and (splits.path == trivial.path).all() and (splits.path == idx.path).all()
    d = pd.concat([urlf, splits.drop(columns="path")], axis=1)
    d["width"], d["height"] = idx.width.astype(float), idx.height.astype(float)
    d["aspect"], d["file_kb"] = d.width / d.height, idx.file_bytes.astype(float) / 1024
    for f in ["mean_r", "mean_g", "mean_b", "white_frac", "gray_std", "is_jpeg"]:
        d[f] = trivial[f].astype(float)

    use = (d.dataset == "phish360") & (d.exclude == "")
    tr, va, te = (np.flatnonzero(use & (d.split == s)) for s in ("train", "val", "test"))
    y = d.label_binary.to_numpy()
    train, val, test = (d.iloc[r].reset_index(drop=True) for r in (tr, va, te))
    images = np.load("data/processed/img224_full.npy", mmap_mode="r")
    frozen = np.load("data/processed/emb_resnet18_full.npy")

    scores = {k: [] for k in ["trivial_floor", "url_only:full", "url_only:no_suspected_artifacts", "image_only",
                              "early_fusion_frozen:full", "late_fusion:full", "late_fusion:no_suspected_artifacts"]}
    for seed in SEEDS:
        # trivial floor: the Phase 1 trivial features and LightGBM settings, trained on train only;
        # row subsampling (0.8) added so the three seeds give genuinely different models

        set_seed(seed)
        floor = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                   random_state=seed, verbose=-1).fit(train[TRIVIAL_FEATURES], train.label_binary)
        scores["trivial_floor"].append(floor.predict_proba(test[TRIVIAL_FEATURES])[:, 1])

        emb_all, prob_all = image_probs_all_rows(seed, images)
        scores["image_only"].append(prob_all[te])

        for feat_set in ("full", "no_suspected_artifacts"):
            drop = ABLATIONS[feat_set]
            num = [f for f in NUMERIC_FEATURES if f not in drop]
            cat = [f for f in CATEGORICAL_FEATURES if f not in drop]
            set_seed(seed)
            lgbm, _ = fit_lgbm(train, val, num, cat, seed)
            url_te = lgbm_predict(lgbm, test, num, cat, train)
            scores[f"url_only:{feat_set}"].append(url_te)
            scores[f"late_fusion:{feat_set}"].append((url_te + prob_all[te]) / 2)

            if feat_set == "full":
                pre = make_preprocessor(num, cat).fit(train[num + cat])
                U = {k: pre.transform(df[num + cat]) for k, df in (("tr", train), ("va", val), ("te", test))}
                mu, sd = frozen[tr].mean(0), frozen[tr].std(0) + 1e-6
                I = {k: (frozen[r] - mu) / sd for k, r in (("tr", tr), ("va", va), ("te", te))}
                model = fit_early(np.hstack([I["tr"], U["tr"]]), y[tr], np.hstack([I["va"], U["va"]]), y[va], seed)
                scores["early_fusion_frozen:full"].append(predict_early(model, np.hstack([I["te"], U["te"]])))
        print(f"seed {seed}: test scores ready", flush=True)
    return y[te], test.path.to_numpy(), {k: np.vstack(v) for k, v in scores.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y, paths, scores = build_test_scores()

    # per-page scores (seed means) for error analysis in Phase 7
    pd.DataFrame({"path": paths, "label_binary": y, **{f"score_{k}": v.mean(0) for k, v in scores.items()}}
                 ).to_csv(OUT / "test_predictions.csv", index=False)

    # shortcut-hard subset: where the trivial model (seed-averaged) is wrong at threshold 0.5
    floor_pred = (scores["trivial_floor"].mean(0) >= 0.5).astype(int)
    hard = floor_pred != y

    for name, mask, path in [("all", np.ones_like(y, bool), Path("results/final_in_distribution.csv")),
                             ("shortcut_hard", hard, OUT / "final_shortcut_hard.csv")]:
        rows = []
        for model, s in scores.items():
            m, feats = (model.split(":") + ["-"])[:2]
            rows.append({"model": m, "url_features": feats, "test_subset": name, "n": int(mask.sum()),
                         "n_phish": int(y[mask].sum()), **evaluate_with_ci(y[mask], s[:, mask])})
        table = pd.DataFrame(rows)
        table.to_csv(path, index=False)
        print(f"\n== {name} (n={mask.sum()}, phishing={y[mask].sum()}) ==")
        print(table[["model", "url_features", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
                     "tpr_at_1pct_fpr", "tpr_at_1pct_fpr_ci_low", "tpr_at_1pct_fpr_ci_high", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
