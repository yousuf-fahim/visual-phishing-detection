# Phase 5 plan: final test and generalization

The Phish360 test split is opened exactly once, by `src/evaluate.py`. Every model was fixed before this point (Q11).

## src/evaluate.py: in-distribution (Phish360 test)
- Models: trivial-feature floor (LightGBM on Phase 1 statistics), URL-only LightGBM (full and no-artifact features),
  image-only fine-tuned MobileNetV3, late fusion (full and no-artifact), early fusion with frozen ResNet18.
  All trained on train, early-stopped on val, seeds 42/43/44 (refits are deterministic).
- Metrics: precision, recall, F1, ROC-AUC, TPR@1%FPR, as the mean over seeds, with 95% bootstrap CIs
  (1,000 resamples of the test set, metric averaged over seeds inside each resample).
- Shortcut-hard subset: test pages the trivial-feature model gets wrong. These are the busy phishing pages and sparse legitimate pages,
  where page density points the wrong way.

## src/generalization.py
1. **Cross-dataset (image-only, fine-tuned MobileNetV3, 3 seeds each direction):**
   Phish360 to Phish-IRIS test, and Phish-IRIS (trained via `train_image --dataset phish-iris`) to Phish360 test,
   next to each model's in-distribution score. Cross-dataset near-duplicates (`cross_dataset_dup`) are dropped from shifted test sets.
2. **Cross-language (Phish360):** language from page text (`src/page_language.py`). URL LightGBM, image MobileNetV3
   (`train_image --subset english`) and their late fusion, trained on English train pages only, then tested on English test pages vs non-English
   test pages, and per language where a language has at least 20 pages of each class.
3. **Cross-time:** Phish360 has no per-sample dates, so this is skipped. The only time contrast is dataset-level
   (Phish-IRIS 2018 vs Phish360 2020-23), covered by step 1.
4. Plot in-distribution vs shifted scores per model.
