# Phase 5: Final test and generalization (75 min)

Run after /clear. Paste everything below the line.

You should see: one final results table, plus cross-dataset, cross-language and cross-time results.
Checkpoint: why is a model that scores 99% in-distribution and 70% on newer data a more useful finding than the 99% alone?

---

Phase 5: the final evaluation and the generalization study. This is the first and only time we use the in-distribution test set.

1. Evaluate URL-only, image-only and fused models on the Phish360 test set: precision, recall, F1, ROC-AUC, TPR at 1% FPR, with bootstrapped 95% confidence intervals. Save results/final_in_distribution.csv.
2. Cross-dataset: train image-only on Phish360 and test on the binary-mapped Phish-IRIS (2018), and the reverse. Report the drop in each direction and explain the label-mapping caveat.
3. Cross-language: detect page language (use the dataset's language field if present, otherwise langdetect on extracted page text, treated as untrusted). Train on English pages, test on non-English, and report per-language results where there is enough data.
4. Cross-time: if Phish360 has dates, train on earlier years and test on later years. If not, say so and skip it.
5. Plot the in-distribution vs shifted performance side by side for each model.
6. Write a short plain-language interpretation: where does the model hold up and where does it break? Do not overstate.
7. Write all of it as src/evaluate.py and src/generalization.py so the numbers are reproducible. Append the Phase 5 entry to docs/learning-log.md and commit.
