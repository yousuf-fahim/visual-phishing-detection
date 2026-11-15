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

## Phase 3: Image models (2026-11-15)

**What we did.** Letterboxed every screenshot onto a 224x224 canvas (`src/preprocess_images.py`), then trained: a small CNN from scratch (0.39M parameters), a fully fine-tuned ImageNet MobileNetV3-small (1.5M), and frozen ResNet18 / DeiT-tiny (a small ViT) / MobileNetV3 backbones with small heads. All on CPU, validation set only, 3 seeds for the best configuration (`results/phase3/metrics_summary.csv`).

**Results (validation ROC-AUC).** Fine-tuned MobileNetV3, full page: **0.967 ± 0.003** (TPR@1%FPR 0.57). Frozen ResNet18 + MLP: 0.954. Same MobileNetV3 on the top half only: 0.952. CNN from scratch: 0.939. All clearly beat the trivial-feature floor (0.85), but none beat the URL baseline (0.970).

**Key concept: transfer learning, and why no flips.**
A network pretrained on ImageNet has already learned general visual building blocks (edges, text strokes, shapes, layouts). With only ~7,500 training screenshots, reusing those building blocks beats learning everything from scratch: the pretrained model reached in 3 epochs what the scratch CNN barely reached in 15. *Fine-tuning* updates all weights; a *frozen backbone* only trains a small head on its output vectors, which is 300x faster on a CPU but slightly less accurate. We used only light augmentation (small crops, colour jitter, blur, JPEG re-compression) and **no horizontal flips**: a mirrored login page, with its logo top-right and text running backwards, never occurs in reality, and layout is part of the signal. The learning curves show classic overfitting after epoch 5-8 (training loss keeps falling, validation loss rises), which early stopping handles by keeping the best validation epoch.

**Likely faculty questions.**
1. *Why did the top half do worse than the full page?* Lower half content (form fields, footers, page density) carries signal too. The 0.015 gap is consistent across 3 seeds, so it's not noise.
2. *Why MobileNetV3 instead of ResNet50?* CPU only: MobileNetV3-small fine-tunes in ~17 min. ResNet18 would take over 40 min per run, so it was used frozen instead. That trade-off is recorded in `docs/plans/phase-3.md`.
3. *The image model doesn't beat the URL model. Is the project a failure?* No: RQ1 asks exactly that question, and the answer is a finding. Phase 4 checks whether the two are complementary, and Phase 7 checks what the image model actually relies on, given the homepage-vs-login confound.

## Phase 4: Fusion and ablation (2026-11-15)

**What we did.** Combined URL features and screenshots two ways (`src/train_fusion.py`, `results/phase4/`): *early fusion* concatenates the image embedding with the URL features and trains one MLP; *late fusion* trains the two models separately and averages their probabilities. Validation set, 3 seeds, plus a robustness probe that blanks one input at test time.

**Results (validation, mean ± std).** URL only 0.970 / TPR@1%FPR 0.63; image only 0.967 / 0.57; early fusion 0.977-0.987 / 0.74-0.79; **late fusion 0.989 / 0.81**. Without the suspected URL artifacts, late fusion still reaches 0.984 (URL alone 0.943).

**Key concept: complementary errors, and why stacking needs care.**
Fusion only helps if the two models get *different* examples wrong. Here they clearly do: at a 1% false-alarm budget, catching 81% instead of 63% means many phishing pages with innocent-looking URLs have suspicious-looking screenshots, and vice versa. A subtle trap: early fusion on the *fine-tuned* image model's embeddings did worse than on a *frozen* ImageNet model's. The fine-tuned network had partly memorised its training screenshots, so on training data its embeddings looked unrealistically clean, and the fusion MLP learned to over-trust them. The robustness probe shows the other trade-off: late fusion degrades gracefully to the remaining input when one is missing, while early fusion with a blank screenshot falls to 0.918, below URL-only.

**Likely faculty questions.**
1. *Is multimodal fusion your contribution?* No. Phish360's authors already published a multimodal method (CrossPhire). This is a reproduction-scale ablation; our contribution is the generalization and shortcut analysis.
2. *Why does a simple average beat a learned fusion network?* The learned one is trained on features that look more reliable on training data than they are (the memorisation effect), and averaging has nothing to overfit.
3. *What happens if an attacker serves a blank page to the screenshotter?* Late fusion falls back to the URL score (0.970). Early fusion drops to 0.918. That's an argument for late fusion in a deployed system.

## Phase 5: Final test and generalization (2026-11-16)

**What we did.** Opened the Phish360 test split once (`src/evaluate.py`), with 95% bootstrap confidence intervals. Then tested generalization (`src/generalization.py`): train on one dataset, test on the other; train on English pages, test on other languages. Full write-up: `docs/findings-phase5.md`.

**Results.** Test ROC-AUC: late fusion 0.984 (TPR@1%FPR 0.77), image 0.967 (0.47), URL 0.961 (0.59), trivial floor 0.853. On the 288 "shortcut-hard" pages the image model falls to 0.809. Cross-dataset, the image model drops to 0.931 / 0.910. On non-English pages, an English-trained image model drops to 0.940 while the URL model doesn't drop.

**Key concept: the test set is used once, and confidence intervals.**
Every choice we made (model, threshold, fusion type) was made on validation. If we peeked at the test set and changed something, the test score would stop being an honest estimate of performance on new data. A *bootstrap confidence interval* shows how much a score could move with a different sample of test pages: resample the test set with replacement 1,000 times, recompute the metric each time, and take the middle 95%. Two models whose intervals overlap heavily (image 0.959-0.974 vs URL 0.952-0.969) aren't reliably different, while late fusion (0.978-0.989) clearly beats both.

**Likely faculty questions.**
1. *Your image model scores 0.967. Why should we believe it understands phishing?* We shouldn't fully: on pages where the density shortcut points the wrong way, it drops to 0.809. We report that openly (RQ4).
2. *Why did you not test across time?* Phish360 has no per-sample dates. The only time contrast available is 2018 (Phish-IRIS) vs 2020-23 (Phish360), and that mixes time with different label definitions.
3. *Why does the URL model not suffer on other languages?* URL features are character statistics and words like "login", which barely depend on the page's language. The screenshot is full of language-specific text.

## Phase 6: Brand identification (stretch, 2026-11-16)

**What we did.** On Phish-IRIS, fine-tuned MobileNetV3 to name the imitated brand (14 brands plus "other"), 3 seeds, grouped splits (`src/brand.py`, `results/phase6/`). Open-set test: 3 brands (adobe, alibaba, microsoft, drawn at random with seed 42 beforehand) were removed from training entirely, and the model had to flag them as unknown by low confidence.

**Results (Phish-IRIS test).** Closed set: accuracy 0.872 ± 0.008, macro-F1 0.802 ± 0.026. Unseen-brand rejection: AUROC 0.828 ± 0.013 (mean top-class confidence 0.94 on known brands vs 0.79 on unknown). Apple is the weakest brand (52%, often mistaken for "other"); boa, wellsfargo, facebook and alibaba are near perfect. The brand-vs-domain consistency check was skipped: Phish-IRIS has no URLs.

**Key concept: closed-set vs open-set recognition.**
A normal classifier must pick one of the classes it was trained on, even for a brand it has never seen, so it will confidently call a new brand's page "dropbox" or "other". *Open-set* recognition adds a way to say "none of the above". The simplest is a threshold on the top softmax probability: unfamiliar inputs tend to get lower confidence. Our AUROC of 0.83 means this works better than chance but far from reliably: softmax networks are known to be over-confident on unfamiliar inputs. Better options exist (energy scores, distance in embedding space), but they're beyond this project.

**Likely faculty questions.**
1. *Why is macro-F1 lower than accuracy?* Accuracy is dominated by the big "other" class (208 of 425 test images). Macro-F1 averages over brands equally, so small, hard brands like apple pull it down.
2. *Some brands have only 1-4 test images. Can you trust those cells?* No. The grouped split balanced phishing vs legitimate, not individual brands. Per-brand numbers for amazon, linkedin and chase aren't interpretable, and we say so.
3. *How would you use brand ID in a real detector?* If the page looks like PayPal but the domain isn't PayPal's, it's suspicious. That needs each brand's real domains, which Phish-IRIS doesn't include, so we couldn't test it.

## Phase 7: Explainability and robustness (2026-11-16)

**What we did.** Asked what the screenshot model actually uses (`src/explain.py`): Grad-CAM heatmaps, occlusion maps over all 1,590 test pages, and classifiers that see only the page border or only the centre. Then tested fragility (`src/robustness.py`) under blur, JPEG, brightness, grayscale, inversion and above-the-fold cropping, and wrote `results/error_analysis.md`.

**Results.**
- On phishing pages the model's attention sits on the **empty area below the login form and the footer, not on the brand logo**. Grad-CAM mass in the top 20% (header/logo) is 0.12 vs 0.20 for an even spread, and in the bottom 20% it's 0.25.
- The border alone (a 20-pixel ring) gives ROC-AUC 0.914, about the same as the centre alone (0.911).
- Every confident false positive is a legitimate login page, and every confident false negative is a phishing clone of a busy homepage.
- Robust to JPEG (0.960 at quality 10) and grayscale (0.961); fragile to heavy blur (0.79), wash-out (0.88) and losing the lower page (0.74 with only the top 20%). Late fusion stays at 0.95 or above in every case.

**Key concept: explanation methods and what they can't prove.**
*Grad-CAM* asks "which regions of the last convolutional layer pushed the score up?" and paints them as a heatmap. *Occlusion* is more direct: hide one patch and see if the answer changes. Both are approximate: Grad-CAM is coarse (7x7 cells here), and occlusion misses evidence that is spread out redundantly (which is why legitimate pages barely react to any single patch). But when several methods, the error analysis and the shortcut test all point the same way, the story is credible. Here that story is: **the model mostly detects "sparse login-style page" rather than "fake brand"**. That's a property of how Phish360 was assembled (legitimate = homepages), not of phishing itself.

**Likely faculty questions.**
1. *Does the model look at the logo, like a human would?* Mostly not. The header region gets less attention than average on phishing pages, and the confident errors are explained by page sparsity, not by logos.
2. *Why does the border alone work so well?* Because the signal is global (how full, colourful and busy the whole page is), and even a thin frame reveals that. A model reading a specific local cue would collapse to near chance on the border.
3. *What would fix this?* Training data with legitimate *login* pages (the real PayPal or bank sign-in pages), so "sparse form" stops predicting the label, plus evaluation on such pages. Our datasets don't provide them.

## Phase 8: Static browser demo (2026-11-16)

**What we did.** Exported the fine-tuned MobileNetV3 to ONNX (`src/export_onnx.py`, checked against PyTorch: max difference 5e-7) and built a static page (`web/`) that scores a dropped screenshot entirely in the browser with onnxruntime-web (WebAssembly). Int8 quantisation was tried and rejected: the file shrank from 6.1 to 1.7 MB, but validation ROC-AUC collapsed from 0.964 to 0.633 (`results/phase8/export_check.json`). Deploy steps for Cloudflare Pages and GitHub Pages are in `docs/DEPLOY.md`.

**Key concept: train/serve skew, and how we ruled it out.**
A model only works if it receives inputs prepared exactly as in training. Browsers resize images with their own filters, so a "224x224 version" made with a canvas would differ slightly from Pillow's Lanczos result: different pixels, silently different scores. We ported Pillow's resampling algorithm line by line to JavaScript (`web/src/preprocess.js`) and tested it: on 23 images (including odd sizes) the JS output is **bit-identical** to Python's, and probabilities agree to within 1e-6 (`results/phase8/parity_test.txt`). An end-to-end self-test in real headless Edge and Chrome (browser image decoder + JS + wasm model) reproduces Python's probability for a bundled synthetic image (`results/phase8/browser_selftest.txt`). The page's Content-Security-Policy only allows same-origin requests, so the screenshot physically can't be sent anywhere.

**Likely faculty questions.**
1. *Why not quantise the model for the web?* We tried: dynamic int8 quantisation destroyed accuracy (AUC 0.633), probably because MobileNetV3's hard-swish activations and squeeze-excite layers are sensitive to it. 6 MB in fp32 loads quickly enough.
2. *How do you know the browser gives the same answer as your Python results?* Bit-exact preprocessing test on 23 images, plus a headless-browser self-test in two browsers.
3. *Is this a security tool?* No. The page says so at the top, shows the known failure modes with their numbers, and a high score explicitly means "looks like the training phishing pages", not "is malicious".

## Phase 9: Report and viva preparation (2026-11-16)

**What we did.** Wrote `docs/report.md` (8-page PDF in `docs/pdf/report.pdf`), the viva cheat sheet with 20 questions and a 5-slide outline (`docs/viva-cheatsheet.md`), a new README with reproduction commands, and `run_all.sh` to regenerate every result. Related work cites only sources we checked: the Phish360/CrossPhire paper (Applied Sciences 2026), the Phish-IRIS paper (ISMSIT 2018), VisualPhishNet, Phishpedia, and the method papers.

**Why.** Every number in the report must trace back to a file in `results/`. While writing, two numbers had come from one-off commands (the English share and the error counts), so we added them to `src/generalization.py` and `src/explain.py`. Re-running those scripts reproduced all existing results byte for byte. The check also exposed a figure bug (the Phish360 shortcut chart had silently included Phish-IRIS after it was added), which is now fixed.

**Key concept: reproducibility as a habit.** Fixed seeds, pinned library versions, scripts instead of notebooks, and results files instead of copied numbers mean anyone, including you in six months, can regenerate every table. "It runs again and gives identical bytes" is the strongest evidence that a number is real.

**Likely faculty questions.**
1. *Can I reproduce your numbers?* Yes: download the two datasets, install `requirements.txt`, and run `bash run_all.sh` (about 4-5 hours on CPU). Seeds are fixed, and spot re-runs gave byte-identical results.
2. *Which of your results would you trust least?* The per-language and per-brand numbers (small samples, wide CIs), and anything on Phish-IRIS, given its format shortcut and the label-mapping assumption.
3. *What is the single most important finding?* The screenshot model's in-distribution score overstates what it understands. On shortcut-hard pages it falls from 0.967 to 0.809, and its errors are legitimate login pages and homepage clones.
