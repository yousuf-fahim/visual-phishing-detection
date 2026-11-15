# Visual Phishing Detection: a Generalization and Shortcut Study

Neural Network lab project (CSE, BUBT). Can a CNN detect phishing pages from screenshots, how does it compare with
URL features and their fusion, does it hold up on another dataset or another language, and **what is it actually
looking at?**

Full write-up: [docs/report.md](docs/report.md). Printable PDFs in [docs/pdf/](docs/pdf/): report, slides, viva cheat sheet, learning log, decisions, generalization findings.
Viva notes: [docs/viva-cheatsheet.md](docs/viva-cheatsheet.md) and [docs/learning-log.md](docs/learning-log.md).

> Research prototype, not a security product. Phish360's authors have published their own multimodal detector
> (CrossPhire). Our contribution is the generalization and shortcut analysis, not a new architecture.

## Research questions

- **RQ1:** Does a screenshot classifier beat URL features, and does fusing both help?
- **RQ2:** Does it hold up across datasets and eras (Phish-IRIS 2018 vs Phish360 2020-2023)?
- **RQ3:** Does it generalize across page languages (trained on English, tested on non-English)?
- **RQ4:** What visual cues does it rely on, and does it use shortcuts?
- **RQ5 (stretch):** Brand identification with unknown-brand rejection on Phish-IRIS.

## Headline results

Phish360 test split (used once), mean of 3 seeds, 95% bootstrap CI ([results/final_in_distribution.csv](results/final_in_distribution.csv)):

| Model | ROC-AUC | TPR @ 1% FPR | F1 |
|---|---|---|---|
| Trivial image statistics only (shortcut floor) | 0.853 [0.832, 0.874] | 0.190 | 0.746 |
| URL features, LightGBM | 0.961 [0.952, 0.969] | 0.587 | 0.855 |
| Screenshot, MobileNetV3-small fine-tuned | 0.967 [0.959, 0.974] | 0.472 | 0.892 |
| Early fusion (frozen ResNet18 + URL, MLP) | 0.983 [0.977, 0.988] | 0.705 | 0.929 |
| **Late fusion (average of URL and screenshot)** | **0.984 [0.978, 0.989]** | **0.770** | 0.929 |

What happens to the screenshot model under shift ([results/phase5/](results/phase5/)):

| Condition | ROC-AUC | TPR @ 1% FPR |
|---|---|---|
| In distribution (Phish360 test) | 0.967 | 0.472 |
| Shortcut-hard pages (where trivial stats are wrong, n = 288) | **0.809** | 0.141 |
| Trained on Phish360, tested on Phish-IRIS (2018) | 0.931 | 0.250 |
| Trained on Phish-IRIS, tested on Phish360 | 0.910 | 0.267 |
| Trained on English pages, tested on non-English pages | 0.940 (vs 0.964 on English) | 0.274 (vs 0.533) |

Main findings:
1. **Fusion helps.** URL and screenshot models make different mistakes.
2. **The screenshot model relies largely on a shortcut.** Phish360's legitimate pages are busy homepages and its phishing pages are sparse login forms. Trivial statistics already reach 0.853, and Grad-CAM shows attention on the empty space below forms rather than on logos. False positives are real login pages, and false negatives are homepage clones ([results/error_analysis.md](results/error_analysis.md)).
3. **Under dataset or language shift, the screenshot model keeps its ranking quality but loses about half its strict-threshold detection rate.** URL features and fusion are more stable.

Brand identification on Phish-IRIS: 15-class accuracy 0.872, macro-F1 0.802; unseen-brand rejection AUROC 0.828 ([results/phase6/summary.json](results/phase6/summary.json)).

## Demo

**Try it: https://yousuf-fahim.github.io/visual-phishing-detection/** (drag a screenshot onto the page).

A static, in-browser demo: the screenshot never leaves the device. Its JavaScript preprocessing is bit-identical
to training. Source in [web/](web/), deploy steps in [docs/DEPLOY.md](docs/DEPLOY.md).

A quick live check on genuine websites ([results/phase8/legit_pages_scores.csv](results/phase8/legit_pages_scores.csv)):
content-rich pages (Wikipedia in English and Bengali, BBC News, python.org, the BUBT homepage) score 0%, but the
**real login pages of GitHub, Google, Microsoft, PayPal and Facebook all score 97.6-99.95% "phishing-like"**.
That is the report's shortcut finding in action.

## How to reproduce

The datasets are **not** included and must not be redistributed. Download them yourself:

- **Phish360:** https://web.cs.hacettepe.edu.tr/~selman/phish360-dataset/ (password-protected zip; request access from the authors).
  Extract so that `data/raw/phish360/trainval/<ID>_<class>/{Label,URL,RAW-HTML,SCREEN-SHOT}/...` and `data/raw/phish360/test/...` exist.
- **Phish-IRIS:** https://web.cs.hacettepe.edu.tr/~selman/phish-iris-dataset/ (academic use only; we used the Kaggle mirror
  `saurabhshahane/phishiris`). Arrange it as `data/raw/phish-iris/train/<class>/*` and `data/raw/phish-iris/val/<class>/*`.

Environment (Python 3.13 tested, CPU only; no GPU needed):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m src.check_env                        # prints versions, writes results/env.json
```

Run everything (about 4-5 hours on a 6-core CPU; each step is a separate module you can also run alone):

```bash
bash run_all.sh
```

The order is the pipeline order: Phase 1 data (`build_index`, `image_stats`, `dedup`, `shortcut_check`, `make_splits`), Phase 2 URL
(`url_features`, `train_url`), Phase 3 images (`preprocess_images`, `embed_images`, `train_frozen`, `train_image`, `summarize_phase3`),
Phase 4 `train_fusion`, Phase 5 (`page_language`, extra `train_image` runs, `evaluate`, `generalization`), Phase 6 `brand`,
Phase 7 (`explain`, `robustness`), Phase 8 (`export_onnx`, then `npm test` / `npm run build` in `web/`). Seeds are fixed (42, 43, 44).

## Repository map

| Path | Contents |
|---|---|
| `src/` | All code. One module per step, each documented at the top |
| `results/` | Every number and figure in the report (CSV/JSON/PNG). Per-screenshot images are git-ignored |
| `docs/report.md` | Lab report. `docs/viva-cheatsheet.md`: 20 Q&As and a slide outline |
| `docs/learning-log.md` | Per-phase notes: what, why, key concept, likely faculty questions |
| `docs/QUESTIONS.md` | Decisions made during the unattended run, with how to change each |
| `docs/plans/` | Phase plans. `docs/findings-phase5.md`: generalization write-up |
| `web/` | Static browser demo (`npm test`, `npm run build`) |
| `data/`, `models/` | Git-ignored: raw data, processed arrays, checkpoints |

## Datasets and credits

- Phish360 and CrossPhire: A. H. A. Almakhamreh, A. S. Bozkir, *Applied Sciences* 16(2):751, 2026.
- Phish-IRIS: F. C. Dalgic, A. S. Bozkir, M. Aydos, "Phish-IRIS: A New Approach for Vision Based Brand Prediction of Phishing
  Web Pages via Compact Visual Descriptors", ISMSIT 2018 (arXiv:1905.07767). Academic use only.
- Pretrained backbones from `timm` (ImageNet weights). Browser inference with `onnxruntime-web`.

## Ethics

Only the downloaded datasets were used. No live phishing URL was visited, fetched or rendered. URLs and HTML were treated
as untrusted text, and no raw screenshots, URLs or HTML are published in this repository.
