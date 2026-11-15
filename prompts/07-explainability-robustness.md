# Phase 7: Explainability and robustness (60 min)

Run after /clear. Paste everything below the line.

Checkpoint: what does Grad-CAM show, and what would it look like if the model were using a shortcut?

---

Phase 7: what is the model actually looking at, and how fragile is it?

1. Grad-CAM on the best image model for about 10 correctly and 10 wrongly classified screenshots. Describe in plain words what it focuses on (logo, login form, header, page borders). Keep outputs in results/ and do not commit raw screenshots.
2. Occlusion sensitivity: slide a masking patch over the screenshot and map how the phishing probability changes. Summarize across the test set: which regions matter most on average?
3. Shortcut check again: classify using only the border pixels, and using only the center, and compare with the metadata-only baseline from Phase 1. Report what this suggests.
4. Perturbations at 4-5 severity levels: blur, JPEG compression, brightness, grayscale, color inversion, and cropping to above-the-fold. Plot accuracy and TPR at 1% FPR against severity.
5. Error analysis: the top false positives and false negatives, with a plausible reason for each (for example a legitimate login page that looks like a bank).
6. Write src/explain.py and src/robustness.py, save outputs to results/, write results/error_analysis.md, append the Phase 7 entry to docs/learning-log.md and commit.
