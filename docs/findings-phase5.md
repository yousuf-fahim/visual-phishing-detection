# Phase 5 findings: where the detector holds up and where it breaks

All numbers come from `results/final_in_distribution.csv`, `results/phase5/final_shortcut_hard.csv`,
`results/phase5/cross_dataset.csv` and `results/phase5/cross_language.csv` (test split, mean of 3 seeds, 95% bootstrap CIs).
The Phish360 test split was used once, here.

## In distribution (Phish360 test, n = 1,590)
- Late fusion is best: ROC-AUC 0.984 (0.978-0.989), catching 77% of phishing at a 1% false-alarm rate. URL-only reaches 0.961 (TPR 0.59) and image-only 0.967 (TPR 0.47). The validation ranking held.
- Every model is far above the trivial-feature floor (0.853), but the floor is high. Page density alone separates the classes a lot.

## Shortcut-hard pages (n = 288: busy phishing pages and sparse legitimate pages)
- **The image model drops from 0.967 to 0.809.** It partly relies on "sparse page means phishing". The URL model drops less (0.927) because it never sees the page. Fusion lands in between (0.907).
- This is the clearest evidence for RQ4. It also means headline screenshot scores on Phish360 overstate how well the model judges *look-alike* pages.

## Across datasets and time (image-only)
- Phish360 model on Phish-IRIS (2018): 0.967 drops to **0.931**. Phish-IRIS model on Phish360 (2020-23): 0.959 (its own test) drops to **0.910**.
- Ranking quality mostly survives, but at the strict 1% false-alarm point roughly half the detection rate is lost (e.g. TPR 0.47 to 0.25).
- Removing the 97 shared near-duplicate clusters hardly changes the numbers, so the transfer isn't just memorised kits.
- Caveat: the two datasets define "legitimate" differently (Phish360 homepages from a top-site list; Phish-IRIS "other" pages, mostly JPEG), and Phish-IRIS still carries a JPEG/colour shortcut after cropping (Q10). Some of the drop is label-definition mismatch, not just time.
- **Cross-time within Phish360 was skipped**: the dataset has no per-sample dates. The only time contrast is 2018 vs 2020-23 at dataset level.

## Across languages (Phish360, models trained on English pages only)
- Image-only: 0.964 on English, **0.940 on non-English**, and TPR@1%FPR halves (0.53 to 0.27). The visual model partly learned English-specific cues (text shapes, typical layouts).
- URL-only does not drop (0.946 vs 0.951): URLs are mostly language-neutral.
- Late fusion stays at 0.978-0.980 on both, because the URL half covers for the image half.
- Training on all languages brings image-only on non-English back to 0.962. Per-language results (French n=85, Spanish n=70) have wide CIs and shouldn't be over-read.

## In one sentence
The screenshot model is strong in distribution and complements URL features well, but part of its skill is a page-density shortcut, and it loses about half its strict-threshold detection rate when the dataset or language changes. Fusing with URL features is the most robust option tested.
