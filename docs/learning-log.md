# Learning log

One entry per phase, written by Claude Code: what we did, why, the key concept in plain language, and three likely faculty questions with short answers. Read it top to bottom before your viva.

## Phase 0: Environment setup (2026-11-15)

**What we did.** Created a Python virtual environment (`.venv`, Python 3.13.3), installed and pinned 19 libraries in `requirements.txt`, set up offline domain parsing, recorded the environment in `results/env.json`, and initialised git.

**Why.** Reproducibility: anyone (including a faculty examiner) can rebuild exactly the same environment from the pinned versions, and every later number can be traced to known library versions.

**Key concept: reproducible environment, offline-only tooling.**
A virtual environment is a private copy of Python plus its libraries for one project, so the project can't break other projects (or be broken by them).
Pinning versions (`torch==2.15.0` instead of `torch`) matters because library updates can change default settings and shift results slightly.
Our machine has no NVIDIA GPU, so PyTorch runs on the CPU. Training a CNN from scratch on CPU is slow, which is why the plan leans on *pretrained* backbones: a network already trained on millions of everyday images, which we either fine-tune briefly or use as a fixed feature extractor.
`tldextract` turns a URL into its *registered domain* (`login.paypal.example.co.uk` gives `example.co.uk`). We configured it to use the bundled suffix list so it never goes online, because of our rule that the project never contacts anything related to phishing URLs.

**Likely faculty questions.**
1. *Why pin library versions?* So results are reproducible. A newer version can change defaults (augmentations, initialisation) and give different numbers.
2. *You have no GPU. Is that a problem?* Not a big one. Pretrained backbones at 224x224 with frozen or briefly fine-tuned layers train in minutes to tens of minutes on a CPU. We note any compromises (smaller images, fewer epochs) explicitly.
3. *Why must URL parsing be offline?* The datasets contain live phishing URLs. Our ethics rule is never to contact them, and offline parsing guarantees we only treat them as text.

## Phase 1: Data audit, shortcut check and splits (2026-11-15, Phish360 only; Phish-IRIS pending)

**What we did.** Indexed all 10,748 Phish360 samples (6,416 legitimate, 4,332 phishing, 46 brands), checked every file, measured trivial image statistics, found near-duplicate screenshots with perceptual hashing, and built grouped 70/15/15 splits (`src/build_index.py`, `image_stats.py`, `dedup.py`, `shortcut_check.py`, `make_splits.py`).

**Why.** A model's score only means something if the test set holds pages it has truly never seen, and if the label can't be guessed from something trivial.

**Key concept: leakage and shortcuts.**
*Leakage:* phishing kits are copy-pasted onto many sites, so the same page shows up dozens of times. With a random split, a copy lands in training and another in test, and the model "recognises" it rather than detecting it. We grouped samples by near-identical screenshot (pHash within 6 of 64 bits) and by registered domain, and kept each group in one split. Even Phish360's official split leaks: 15.5% of its test samples have a relative in trainval.
*Shortcut:* a model takes the easiest signal that separates the labels. Here, legitimate samples are busy homepages and phishing samples are sparse login forms, so file size and whiteness alone reach ROC-AUC 0.85 without seeing any "phishing" cue. Image dimensions carry no signal (AUC 0.50), since almost all are 1280x960. We also removed 116 blank screenshots (110 phishing), because "blank means phishing" is pure shortcut.

**Likely faculty questions.**
1. *Why not a random split?* Near-duplicate copies would sit on both sides and inflate test scores. Grouping by pHash cluster and domain gives zero groups across splits (`results/phase1/splits_summary.json`).
2. *Your images could be classified by file size. Doesn't that invalidate the study?* It sets a floor: trivial features reach AUC 0.85, so an image model is only interesting if it clearly beats that, especially on the "shortcut-hard" subset. Measuring this is itself part of RQ4.
3. *What is a perceptual hash?* A 64-bit fingerprint of an image's coarse brightness pattern. Visually similar images get fingerprints differing in only a few bits, unlike a cryptographic hash, where one pixel changes everything.
