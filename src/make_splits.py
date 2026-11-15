"""Phase 1, step 6: reproducible grouped train/val/test splits (70/15/15).

Run: .venv/Scripts/python -m src.make_splits   (after build_index, image_stats, dedup)

How: StratifiedGroupKFold cuts each dataset into 20 folds of ~5% each, keeping
every leakage group (near-duplicate screenshots + same registered domain) inside
ONE fold and keeping the phishing/legit ratio similar in every fold.
Folds 0-2 -> test (15%), 3-5 -> val (15%), 6-19 -> train (70%). Seed 42.

Exclusions (column `exclude`, PROVISIONAL, see docs/QUESTIONS.md):
  near_blank  screenshot is almost all white or a flat colour (white_frac > 0.97 or
              grey std < 5). These are failed or taken-down page loads: no visual
              evidence, and 110 of 116 are labelled phishing, a pure shortcut.
Excluded rows still get a split (so the grouping stays intact) but every phase
drops them, so all models are compared on exactly the same samples.

Outputs
  data/processed/splits.csv            path, dataset, split, group_id, label_binary, exclude
  results/phase1/splits_summary.json   sizes, label balance, and the no-overlap check
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_FOLDS = 20
TEST_FOLDS, VAL_FOLDS = {0, 1, 2}, {3, 4, 5}


def assign_splits(d: pd.DataFrame) -> pd.Series:
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    split = pd.Series("train", index=d.index)
    for fold, (_, members) in enumerate(cv.split(d, d.label_binary, d.group_id)):
        if fold in TEST_FOLDS:
            split.iloc[members] = "test"
        elif fold in VAL_FOLDS:
            split.iloc[members] = "val"
    return split


def main() -> None:
    idx = pd.read_csv("data/processed/index.csv", keep_default_na=False)
    feats = pd.read_csv("data/processed/image_features.csv", keep_default_na=False)
    groups = pd.read_csv("data/processed/groups.csv")
    df = idx.merge(feats, on="path").merge(groups, on="path")

    near_blank = (df.white_frac.astype(float) > 0.97) | (df.gray_std.astype(float) < 5)
    df["exclude"] = ""
    df.loc[near_blank, "exclude"] = "near_blank"

    df["split"] = ""
    for name, d in df.groupby("dataset"):
        df.loc[d.index, "split"] = assign_splits(d.reset_index(drop=True)).to_numpy()

    out = df[["path", "dataset", "split", "group_id", "label_binary", "exclude"]]
    out.to_csv("data/processed/splits.csv", index=False)

    # ---- checks and summary ----
    groups_per_split = df.groupby("group_id").split.nunique()
    assert (groups_per_split == 1).all(), "a leakage group crosses splits!"
    domains_in_two_splits = int((df.groupby(["dataset", "registered_domain"]).split.nunique() > 1).sum())
    used = df[df.exclude == ""]
    summary = {
        "seed": SEED,
        "groups_crossing_splits": int((groups_per_split > 1).sum()),
        "registered_domains_crossing_splits": domains_in_two_splits,
        "excluded": df.exclude.value_counts().drop("", errors="ignore").to_dict(),
        "excluded_by_label": {("phish" if k else "legit"): int(v)
                              for k, v in df[df.exclude != ""].label_binary.value_counts().items()},
        "after_exclusion": {
            name: {s: {"n": int(len(x)), "share": round(len(x) / len(d), 4),
                       "phishing_fraction": round(x.label_binary.mean(), 4)}
                   for s, x in d.groupby("split")}
            for name, d in used.groupby("dataset")},
    }
    Path("results/phase1").mkdir(parents=True, exist_ok=True)
    Path("results/phase1/splits_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
