# Phase 6: Brand identification (60 min) OPTIONAL STRETCH

Run after /clear, only if phases 0-5 are healthy. Paste everything below the line.

Checkpoint: why must a real system be able to say "unknown brand" instead of always picking one of the 14?

---

Phase 6: which brand is being imitated? Use Phish-IRIS and its grouped splits.

1. Train a multi-class classifier over the brand classes plus "other" using transfer learning (the best backbone from Phase 3). Report accuracy, macro F1 and a confusion matrix, and compare with published baselines mentioned in PLAN.md only if we can cite them, otherwise skip the comparison.
2. Open-set test: hold out 2-3 brands during training and treat them as "unknown". Use a confidence threshold (max softmax) to reject unknowns, and report AUROC for separating known from unknown brands.
3. If the data lists each brand's legitimate domain, add a consistency check: flag when the predicted brand does not match the page's domain. If the data does not include this, skip it and say so.
4. Write it as src/brand.py with fixed seeds, save results to results/, append the Phase 6 entry to docs/learning-log.md and commit.
