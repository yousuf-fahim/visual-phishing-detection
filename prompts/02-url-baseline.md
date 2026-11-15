# Phase 2: URL-feature baseline (45 min)

Run after /clear. Paste everything below the line.

Checkpoint: which URL features matter most, and could any of them be a dataset artifact rather than a real phishing signal?

---

Phase 2: URL baseline. This is the non-visual comparison point, and it builds on my earlier phishing URL lab.

1. In src/url_features.py, extract lexical features from the URL strings only (never fetch the URL): URL and hostname length, digit and special-character ratios, number of subdomains, hyphens, presence of '@', IP address as host, https flag, entropy, path depth, query length, suspicious keywords, TLD and registered domain. Treat URLs as untrusted strings.
2. Train logistic regression, LightGBM and a small PyTorch MLP on Phish360 using the grouped splits. Validation set only.
3. Report precision, recall, F1, ROC-AUC and true-positive rate at 1% false-positive rate.
4. Shortcut check: show the top features, and run an ablation with the suspicious ones removed (for example the https flag or TLD). Say plainly if a feature looks like a dataset artifact.
5. Save the models and metrics under models/ and results/ from src/train_url.py with a fixed seed.
6. Append the Phase 2 entry to docs/learning-log.md and commit.
