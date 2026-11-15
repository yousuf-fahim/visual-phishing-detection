"""Phase 5: generalization tests. Does the detector hold up when the data shifts?

Run: .venv/Scripts/python -m src.generalization
     (after src.evaluate and the Phase 5 train_image runs, see docs/plans/phase-5.md)

1. Cross-dataset (image-only, fine-tuned MobileNetV3, 3 seeds per direction)
     Phish360 (2020-23) model -> Phish-IRIS test (2018), and Phish-IRIS model -> Phish360 test,
     each next to its in-distribution score. Pages whose near-duplicate exists in the other
     dataset (cross_dataset_dup) are removed from the shifted test sets, since they'd be
     "seen" pages, not a real shift. Results with them kept are reported too.
2. Cross-language (Phish360): URL LightGBM, image MobileNetV3 and their late fusion, trained on
     ENGLISH pages only, tested on English vs non-English test pages, and per language where a
     language has >= MIN_PER_CLASS test pages of each class. The all-language models from
     src.evaluate are scored on the same groups for comparison.
3. Cross-time: Phish360 has no per-sample dates, so this test is skipped (see the report).

Outputs: results/phase5/cross_dataset.csv, cross_language.csv, generalization.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import SEEDS, evaluate_with_ci, image_probs_all_rows, lgbm_predict
from src.train_url import fit_lgbm, set_seed
from src.url_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

OUT = Path("results/phase5")
MIN_PER_CLASS = 20
SHOW = ["roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "tpr_at_1pct_fpr", "f1"]


def load():
    splits = pd.read_csv("data/processed/splits.csv", keep_default_na=False)
    urlf = pd.read_csv("data/processed/url_features.csv", keep_default_na=False)
    lang = pd.read_csv("data/processed/languages.csv", keep_default_na=False)
    assert (splits.path == urlf.path).all() and (splits.path == lang.path).all()
    d = pd.concat([urlf, splits.drop(columns="path")], axis=1)
    d["language"] = lang.language
    d["cross_dataset_dup"] = d.cross_dataset_dup.astype(str) == "True"
    return d


def cross_dataset(d: pd.DataFrame, images) -> pd.DataFrame:
    y = d.label_binary.to_numpy()
    ok = d.exclude == ""
    tests = {ds: ok & (d.dataset == ds) & (d.split == "test") for ds in ("phish360", "phish-iris")}
    models = {"phish360": "", "phish-iris": "phish-iris-all-"}  # checkpoint prefixes
    probs = {ds: np.vstack([image_probs_all_rows(s, images, pre)[1] for s in SEEDS]) for ds, pre in models.items()}
    rows = []
    for train_ds in models:
        for test_ds, mask in tests.items():
            shifted = train_ds != test_ds
            variants = [("in-distribution", mask)] if not shifted else [
                ("cross-dataset (shared near-duplicates removed)", mask & ~d.cross_dataset_dup),
                ("cross-dataset (all test pages)", mask)]
            for cond, m in variants:
                r = np.flatnonzero(m)
                rows.append({"model": "image_only (fine-tuned MobileNetV3)", "train_dataset": train_ds,
                             "test_dataset": test_ds, "condition": cond, "n": len(r), "n_phish": int(y[r].sum()),
                             **evaluate_with_ci(y[r], probs[train_ds][:, r])})
    return pd.DataFrame(rows)


def cross_language(d: pd.DataFrame, images) -> pd.DataFrame:
    y = d.label_binary.to_numpy()
    p360 = (d.dataset == "phish360") & (d.exclude == "")
    en = d.language == "en"
    tr, va = np.flatnonzero(p360 & en & (d.split == "train")), np.flatnonzero(p360 & en & (d.split == "val"))
    test = p360 & (d.split == "test")
    train_df, val_df = d.iloc[tr].reset_index(drop=True), d.iloc[va].reset_index(drop=True)
    num, cat = NUMERIC_FEATURES, CATEGORICAL_FEATURES

    # Scores on ALL rows from English-only models and from all-language models (same recipe, all train pages)
    tr_all, va_all = np.flatnonzero(p360 & (d.split == "train")), np.flatnonzero(p360 & (d.split == "val"))
    train_all, val_all = d.iloc[tr_all].reset_index(drop=True), d.iloc[va_all].reset_index(drop=True)
    eng_url, eng_img, all_url = [], [], []
    for s in SEEDS:
        set_seed(s)
        lgbm, _ = fit_lgbm(train_df, val_df, num, cat, s)
        eng_url.append(lgbm_predict(lgbm, d, num, cat, train_df))
        eng_img.append(image_probs_all_rows(s, images, "phish360-english-")[1])
        set_seed(s)
        lgbm_all, _ = fit_lgbm(train_all, val_all, num, cat, s)
        all_url.append(lgbm_predict(lgbm_all, d, num, cat, train_all))
    eng_url, eng_img, all_url = np.vstack(eng_url), np.vstack(eng_img), np.vstack(all_url)
    all_img = np.vstack([image_probs_all_rows(s, images)[1] for s in SEEDS])

    groups = {"English": test & en, "non-English": test & ~en & (d.language != "unknown")}
    vc = d[test].language.value_counts()
    for lang in vc.index:
        m = test & (d.language == lang)
        if lang not in ("en", "unknown") and y[m].sum() >= MIN_PER_CLASS and (1 - y[m]).sum() >= MIN_PER_CLASS:
            groups[f"lang={lang}"] = m

    rows = []
    for gname, m in groups.items():
        r = np.flatnonzero(m)
        url_all = all_url[:, r]
        for trained_on, model, s in [
            ("English only", "url_only", eng_url[:, r]),
            ("English only", "image_only", eng_img[:, r]),
            ("English only", "late_fusion", (eng_url[:, r] + eng_img[:, r]) / 2),
            ("all languages", "url_only", url_all),
            ("all languages", "image_only", all_img[:, r]),
            ("all languages", "late_fusion", (url_all + all_img[:, r]) / 2),
        ]:
            rows.append({"trained_on": trained_on, "model": model, "test_group": gname, "n": len(r),
                         "n_phish": int(y[r].sum()), **evaluate_with_ci(y[r], s)})
    return pd.DataFrame(rows)


def plot(cd: pd.DataFrame, cl: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    # cross-dataset: in-distribution vs shifted per training set
    ax = axes[0]
    keep = cd[cd.condition != "cross-dataset (all test pages)"]
    labels, vals, lo, hi, colors = [], [], [], [], []
    for _, r in keep.iterrows():
        labels.append(f"train {r.train_dataset}\ntest {r.test_dataset}")
        vals.append(r.roc_auc)
        lo.append(r.roc_auc - r.roc_auc_ci_low)
        hi.append(r.roc_auc_ci_high - r.roc_auc)
        colors.append("#4C72B0" if r.condition == "in-distribution" else "#C44E52")
    ax.bar(labels, vals, yerr=[lo, hi], color=colors, capsize=4)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_title("Cross-dataset, image-only (blue = in-distribution, red = shifted)")
    ax.set_ylabel("test ROC-AUC (95% CI)")
    # cross-language: English vs non-English for English-trained models
    ax = axes[1]
    sub = cl[cl.trained_on == "English only"]
    groups = list(dict.fromkeys(sub.test_group))
    models = ["url_only", "image_only", "late_fusion"]
    w = 0.8 / len(groups)
    for i, g in enumerate(groups):
        s = sub[sub.test_group == g].set_index("model").loc[models]
        ax.bar(np.arange(len(models)) + i * w, s.roc_auc, w,
               yerr=[s.roc_auc - s.roc_auc_ci_low, s.roc_auc_ci_high - s.roc_auc], capsize=3, label=g)
    ax.set_xticks(np.arange(len(models)) + w * (len(groups) - 1) / 2, models)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Cross-language: trained on English pages only")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "generalization.png", dpi=120)


def main() -> None:
    d = load()
    images = np.load("data/processed/img224_full.npy", mmap_mode="r")
    cd = cross_dataset(d, images)
    cd.to_csv(OUT / "cross_dataset.csv", index=False)
    print(cd[["train_dataset", "test_dataset", "condition", "n", "n_phish"] + SHOW].to_string(index=False))
    cl = cross_language(d, images)
    cl.to_csv(OUT / "cross_language.csv", index=False)
    print(cl[["trained_on", "model", "test_group", "n", "n_phish"] + SHOW].to_string(index=False))
    plot(cd, cl)


if __name__ == "__main__":
    main()
