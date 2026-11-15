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

## Phase 2: URL-feature baseline (2026-11-15)

**What we did.** Extracted 29 lexical features from each URL string (lengths, character ratios, subdomains, entropy, suspicious words, TLD, and so on; never fetching anything), and trained logistic regression, LightGBM and a small PyTorch MLP on the grouped train split. Evaluated on validation only, with 3 seeds (`src/url_features.py`, `src/train_url.py`, `results/phase2/`).

**Why.** RQ1 asks whether screenshots beat URLs, so we need a strong, honest URL baseline first.

**Key concept: dataset artifacts in features.**
A feature can predict the label without being a real phishing signal, if it reflects *how the data was collected*. Here the strongest feature is `has_www`: legitimate URLs come from a list of popular sites written as `www.example.com`, while phishing URLs come from phishing feeds. A real attacker can trivially add `www.`, so a model leaning on it would be easy to fool. We tested this with an **ablation**, removing the suspect features and retraining. Result: LightGBM goes from ROC-AUC 0.970 to 0.943, and TPR@1%FPR from 0.63 to 0.53. Some of the score was artifact, most was genuine (suspicious words, dots, hyphens, entropy). `is_https` turned out *not* to be an artifact here (40% vs 43%).

**Likely faculty questions.**
1. *Why is TPR at 1% FPR so much lower than ROC-AUC suggests?* AUC averages over all thresholds. At the strict operating point a real deployment needs (1 legit site in 100 flagged), only about 63% of phishing is caught.
2. *Your top feature is 'has www'. Isn't that suspicious?* Yes, and we say so. Removing it alone costs almost nothing, because correlated features take over. Removing every suspected artifact costs about 0.03 AUC, and we report both versions.
3. *Why use an MLP on tabular features at all?* As a neural baseline for a Neural Network lab. On about 30 tabular features it ties LightGBM (0.971 vs 0.970), which is the usual finding: trees match or beat neural nets on small tabular data.

### Phase 1 addendum: Phish-IRIS added (2026-11-15)

Phish-IRIS (2,849 images, 14 brands plus "other") was indexed and split the same way. Two lessons. First, **trivial features predict Phish-IRIS almost perfectly** (ROC-AUC 0.995): its legitimate images are mostly square JPEGs and its phishing images wide PNGs. Letterboxing keeps the aspect ratio, so the grey band width would still give the label away. Phish-IRIS images are therefore cropped to the same 4:3 browser-viewport shape as Phish360 before resizing. Second, **97 phishing page designs appear in both datasets**, years apart: kits are reused for a long time. Those samples are flagged and kept out of the cross-dataset tests.

*Faculty question: "Isn't Phish-IRIS a standard benchmark? Why distrust it?"* Because a classifier that never looks at page content (only shape, file format and mean colour) already scores 0.995 on it. A high Phish-IRIS score therefore says little about visual phishing cues.
