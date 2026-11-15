# Phase 1 plan: data audit, shortcut check, splits

Status when written: Phish360 is in data/raw/phish360 (10,748 samples). Phish-IRIS has not arrived yet.
Every script handles Phish-IRIS automatically once it's in data/raw/phish-iris, so Phase 1 is re-run then.

## Facts found before planning
- Layout: `{trainval,test}/<ID>_<class>/{Label/label.txt, URL/url.txt, RAW-HTML/index.html, SCREEN-SHOT/screen_shoot.png}`.
- ID prefix L = legitimate, P = phishing. trainval has 46 brands. In test, all phishing is labelled "phish" (no brand).
- label.txt has a UTF-8 BOM, and 85 are empty. **The folder name is the label source.** label.txt is only cross-checked.
- No date or language fields. Language will be detected from page text in Phase 5. Dates exist only at dataset level (Phish360 2020-2023, Phish-IRIS 2018).
- 10 legitimate index.html files went missing during extraction (screenshots intact). Recorded as missing.

## Steps (one script each, all in src/, outputs to data/processed and results/phase1)
1. `build_index.py`: walk both datasets and parse labels, URLs (plain text) and registered domains (offline).
   Record image width, height and file size, and flag corrupt or missing files.
   Writes `data/processed/index.csv`, `label_mapping.json`, `results/phase1/dataset_summary.json`,
   a class-count chart, and a sample grid (in `results/phase1/screenshots/`, git-ignored).
2. `shortcut_check.py`: trivial features (width, height, aspect ratio, file size, mean RGB, white fraction),
   then logistic regression and LightGBM with grouped 5-fold CV. Reports ROC-AUC, F1 and TPR at 1% FPR.
3. `dedup.py`: 64-bit perceptual hash (pHash) per screenshot. Near-duplicates are pairs within Hamming distance 6
   (a ~10% bit difference, a common threshold). They're merged with union-find, together with same-registered-domain samples, into one `group_id`.
   Also reports cross-split overlap of the official Phish360 split, and cross-dataset overlap.
4. `make_splits.py`: grouped, label-stratified 70/15/15 split per dataset (StratifiedGroupKFold, seed 42).
   Writes `data/processed/splits.csv` plus a check that no group crosses splits.
5. Learning-log entry and commit.

## Decision to flag (QUESTIONS.md, PROVISIONAL)
- The official Phish360 test split vs our own grouped split: see QUESTIONS Q4 after the leakage check.
