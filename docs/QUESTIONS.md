# Decisions made while Fahim was away

Each entry: what came up, the options, what I chose, and how to change it.

## Q1 (Phase 0): Python version

- **What came up:** the plan asks for Python 3.10-3.12, but this machine only has 3.13.3.
- **Options:** (a) use 3.13; (b) install 3.12 system-wide. Option (b) changes files outside the project, which AUTOPILOT counts as a blocker.
- **Chose:** (a). Every required library has 3.13 wheels and passed an import check (`results/env.json`).
- **To change:** install Python 3.12 from python.org, delete `.venv`, then run
  `py -3.12 -m venv .venv` and `.venv/Scripts/python -m pip install -r requirements.txt`.

## Q2 (Phase 0): git branches

- **What came up:** your global rules say never commit to `main` and always open a PR. The project CLAUDE.md says to commit at the end of each phase, and Phase 0 says to run `git init`. A new repo has no remote, so no PR is possible, and the first commit has to go on `main`.
- **Chose:** the initial commit goes on `main`. Each later phase gets its own branch (`feat/phase-1-data-audit`, and so on), committed locally. Nothing is pushed or merged, since there's no remote and creating a GitHub repo is your call.
- **To change:** create the GitHub repo, `git remote add origin <url>`, `git push -u origin main`, then push each phase branch and open PRs.

## Q3 (Phase 0): OneDrive (resolved)

- Fahim confirmed this folder is under OneDrive but not synced, so no action is needed.

## Note: OpenCV installed twice

`grad-cam` pulls in `opencv-python-headless` alongside `opencv-python`. Both installed and `import cv2` works. If cv2 ever breaks, uninstall both and reinstall just `opencv-python`.

## Q4 (Phase 1, PROVISIONAL): own grouped split instead of Phish360's official trainval/test

- **What came up:** the official Phish360 test split leaks. 429 of 2,771 test samples (15.5%) share a leakage group with trainval: 80 have a near-duplicate screenshot there and 356 share a registered domain (`results/phase1/leakage_report.json`).
- **Options:** (a) keep the official split (comparable to the paper, but leaky); (b) official test minus the leaked samples; (c) our own grouped 70/15/15 over all 10,748 samples.
- **Chose:** (c), as the Phase 1 prompt asks: `src/make_splits.py`, seed 42, zero groups crossing splits. The official split is kept as the `official_split` column of `index.csv`, so (a) or (b) can be reported as a side result.
- **To change:** in `make_splits.py`, replace `assign_splits` with the official column (mapping trainval to train/val with a grouped cut).

## Q5 (Phase 1, PROVISIONAL): exclude 116 near-blank screenshots

- **What came up:** 116 screenshots are almost entirely white or one flat colour (110 phishing, 6 legitimate): failed or taken-down page loads. They hold no visual phishing evidence, and "blank means phishing" is a shortcut.
- **Chose:** mark them `exclude=near_blank` in `splits.csv`. Every phase drops them, so all models (URL, image, fusion) are compared on identical samples.
- **To change:** loosen or remove the `near_blank` rule in `make_splits.py` and re-run it.

## Q6 (Phase 1, PROVISIONAL): label source

- **Chose:** the sample folder name (`<ID>_legitimate` or `<ID>_<brand>`/`_phish`) is the label. It also agrees with the L/P ID prefix in all 10,748 cases. `label.txt` is unreliable: 86 are empty, and 8 disagree with nonsense values ('com', 'fargo', 'bank'). See `data/processed/label_mapping.json`.
- **To change:** edit `index_phish360()` in `src/build_index.py`.

## Q7 (Phase 1): homepage vs login-page confound (a finding, no action needed)

- Phish360's legitimate samples are mostly busy **homepages**, while phishing samples are sparse **login forms**. Trivial statistics alone (file size, whiteness, mean colour) reach ROC-AUC 0.85 under grouped CV (`results/phase1/shortcut_check.json`). Image size carries no signal (AUC 0.50).
- **Consequence:** a screenshot model may learn "login page vs homepage" instead of "fake vs real". We can't fix this by resizing. Plan: report the 0.85 trivial-feature floor next to every image model, and evaluate on a "shortcut-hard" subset (samples the trivial model gets wrong or is unsure about; scores in `data/processed/shortcut_oof.csv`) in Phases 5 and 7. This is a strong discussion point for RQ4.

## Note: 10 HTML files missing

10 legitimate `RAW-HTML/index.html` files vanished from both the Downloads copy and data/raw shortly after extraction (probably antivirus). Their screenshots, URLs and labels are intact, and they're flagged in `index.csv` (`html_present=False`). Only HTML-based analysis is affected.

## Q8 (Phase 2): URL features that are dataset artifacts (a finding)

- `has_www` is the top LightGBM feature (26% of gain): 71% of legitimate URLs start with `www.`, vs 14% of phishing. That reflects how the legitimate list was collected (top-site domains), not a property of phishing. TLD is skewed the same way (`.cz`, `.de`, `.ro` appear almost only as legitimate), and `is_root_path` is the homepage confound from Q7.
- Removing all of them drops LightGBM from ROC-AUC 0.970 to 0.943, and TPR@1%FPR from 0.63 to 0.53 (`results/phase2/metrics_summary.csv`). The URL baseline stays strong without them.
- **Chose:** keep the full feature set as the main URL baseline (matching the literature), and report the `no_suspected_artifacts` variant next to it in RQ1. The fusion phase uses both variants, so it can be checked for leaning on the artifacts.

## Q9 (Phase 1, PROVISIONAL): Phish-IRIS source and label mapping

- **Source:** the Kaggle mirror `saurabhshahane/phishiris` (Fahim's download), not the authors' Hacettepe/Mendeley page. It has 1,312 train and 1,537 test images, vs 1,313 and 1,539 in the paper (1 and 2 missing). The folder `val` is the official test set. The report cites the original Phish-IRIS paper.
- **Mapping:** the 14 brand classes become phishing (1), and `other` becomes legitimate (0), following the authors' description of `other` as legitimate pages. That gives 1,452 phishing and 1,397 legitimate (`data/processed/label_mapping.json`). No URLs and no HTML, so Phish-IRIS is image-only.
- **To change:** edit `index_phish_iris()` in `src/build_index.py` and re-run Phase 1.

## Q10 (Phase 1/3): Phish-IRIS shape shortcut, so its images are cropped to a 4:3 viewport

- **What came up:** on Phish-IRIS, trivial features reach ROC-AUC **0.995** (TPR@1%FPR 0.88). Most `other` images are square 1024x1024 JPEGs, while brand images are wide PNGs: aspect ratio alone gives AUC 0.80, and file format alone 0.79. Letterboxing removes the size but **not** the shape, because the grey bands' width encodes the aspect ratio.
- **Chose:** Phish-IRIS images are top-cropped to the 4:3 viewport shape of Phish360 screenshots (1280x960) before letterboxing (`src/preprocess_images.py`). Phish360 is unchanged (aspect AUC 0.50 there), so the Phase 2-3 models stay valid. JPEG compression artefacts and colour differences can't be removed this way. Phish-IRIS results must be read with that caveat, and Phase 7 re-tests trivial statistics on the preprocessed inputs.
- **Also found:** 97 near-duplicate clusters are shared between Phish-IRIS (2018) and Phish360 (2020-23), covering 208 and 247 samples, almost all phishing: the same kits persist for years. They're flagged `cross_dataset_dup` in `splits.csv` and dropped from the cross-dataset tests. A third of Phish-IRIS (951 of 2,849) has a near-duplicate within the dataset.

## Q11 (Phase 4): which fused model goes to the final test

- **Chose:** late fusion (plain average of URL LightGBM and fine-tuned MobileNetV3 probabilities). It's best on validation (ROC-AUC 0.989 ± 0.001, TPR@1%FPR 0.81) and has no fitted weights. Early fusion with frozen ResNet18 embeddings (0.987) also goes to Phase 5 as the "learned fusion" representative. Both use the full URL feature set, with the `no_suspected_artifacts` variant reported alongside.
- **To change:** edit the model list in `src/evaluate.py`.
