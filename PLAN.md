# Visual Phishing Detection: Plan

Prepared 15 Nov 2026.

## What you're building

A study of visual phishing detection with neural networks, plus an optional in-browser demo. The question is not just "can a CNN spot phishing screenshots" (published models already reach about 98-99% on standard datasets) but "does it still work on newer data, in other languages, and what is it actually looking at?"

- **Image models:** a small CNN from scratch, a pretrained ResNet or MobileNet, and optionally a small ViT.
- **URL baseline:** lexical URL features with logistic regression and LightGBM (builds on your earlier phishing lab).
- **Fusion:** screenshot embedding plus URL features, with an ablation table.
- **Generalization:** cross-dataset (Phish-IRIS 2018 vs Phish360 2020-2023), cross-language, and cross-time tests.
- **Brand identification (stretch):** which of the 14 brands in Phish-IRIS is being imitated, with unknown-brand rejection.
- **Explainability:** Grad-CAM, occlusion tests, and a shortcut check.

**Final deliverables:** repo, results tables and charts, a 6-8 page report, a viva cheat sheet, and an optional static demo.

**Novelty note:** Phish360's authors have already published a multimodal method on it (CrossPhire). Do not claim multimodal fusion as new. Your contribution is the generalization study and the explanation of what the models rely on.

## Time estimate

| Phase | What happens | Your active time | Notes |
|---|---|---|---|
| 0 | Environment setup | 20-30 min | |
| 1 | Data audit, shortcut check, splits | 75-90 min | Dataset download is manual |
| 2 | URL-feature baseline | 45 min | |
| 3 | Image models | 90-120 min | Training wait; GPU helps |
| 4 | Fusion and ablation | 60 min | |
| 5 | Final test and generalization tests | 75 min | Test set used once here |
| 6 | Brand identification (stretch) | 60 min | Optional |
| 7 | Explainability and robustness | 60 min | |
| 8 | Static browser demo (optional) | 90 min | Optional |
| 9 | Report and viva prep | 90-120 min | |

**Core path (phases 0-5, 7, 9): about 9-10 hours. With brand identification and the demo: about 11-13 hours. Budget 14 for debugging.**

**Suggested split:** Saturday for phases 0-4 (about 5.5 hours), Sunday for phases 5-9 (about 6.5 hours). If you run short, skip 6 and 8.

## Before you start

- Claude Code installed and signed in.
- Python 3.10-3.12, Node 20+ (only if you do the demo), git.
- **Datasets (manual step):** download into `data/raw/`.
  - Phish360: https://web.cs.hacettepe.edu.tr/~selman/phish360-dataset/
  - Phish-IRIS: https://web.cs.hacettepe.edu.tr/~selman/phish-iris-dataset/ (also on Mendeley Data, DOI 10.17632/tk4fkswtj5.1)
  - Optional extra: PhishWeb (Phish-Blitz), public on GitHub.
- A GPU is optional. The plan uses pretrained backbones and 224x224 inputs, so CPU is workable. Claude Code will estimate training time before each run.
- **Handle the data safely:** never open dataset HTML in a browser, and never visit the phishing URLs.

## How to work with Claude Code

- Open a terminal in this folder and run `claude`.
- Paste one prompt file from `prompts/` at a time, in order. Files marked PLAN are best run in plan mode (Shift+Tab).
- Run `/clear` between phases. `CLAUDE.md` and `docs/learning-log.md` carry the context forward.
- Read each new learning-log entry before moving on.
- For unattended runs, see `AUTOPILOT.md`.

## Risks and how the plan handles them

| Risk | Handling |
|---|---|
| Dataset format differs from what the prompts assume | Phase 1 inspects first and adapts, and records the findings |
| Label mapping between Phish-IRIS and binary phishing/legitimate is unclear | Phase 1 documents the mapping and flags it for you to verify |
| Near-duplicate phishing kits leak across splits | Phase 1 groups by domain and screenshot hash, and checks cross-dataset overlap |
| Models learn shortcuts (image size, browser chrome) | Phase 1 and 7 run a metadata-only baseline |
| Language or date metadata is missing | Phase 5 detects language from page text and uses whatever dates exist |
| Training is too slow | Frozen-backbone embeddings, smaller images, fewer epochs |

## Dataset notes

- **Phish360:** about 10,748 samples from 2020-2023, unique (URL, HTML, screenshot) triplets, 30+ languages. Confirm columns, download steps and license on its page.
- **Phish-IRIS:** 15 classes (14 brands plus "other"), 1,313 training and 1,539 testing screenshots, collected March-May 2018. Academic use only. Cite their paper.
- Confirm folder layouts and licenses yourself. Only the dataset pages were checked, not the files.
- Cite every dataset you use in the report.

## Final checklist

- Every number in the report traces to a file in `results/`.
- No raw screenshots or phishing content in the public repo.
- You can explain every item in `docs/viva-cheatsheet.md` in your own words. Read the learning log once top to bottom the night before.
