# Status

Last updated: 2026-11-16 (end of the autopilot run).

## Final summary

**All phases (0-9) are complete**, including the two optional ones (6 brand identification, 8 demo). Nothing is blocked.

### Headline results (all from `results/`)

| | ROC-AUC | TPR @ 1% FPR |
|---|---|---|
| Phish360 test: URL only (LightGBM) | 0.961 | 0.587 |
| Phish360 test: screenshot (MobileNetV3) | 0.967 | 0.472 |
| Phish360 test: **late fusion** | **0.984** | **0.770** |
| Trivial image statistics only (shortcut floor) | 0.853 | 0.190 |
| Screenshot on shortcut-hard pages (n = 288) | **0.809** | 0.141 |
| Screenshot, Phish360 to Phish-IRIS / Phish-IRIS to Phish360 | 0.931 / 0.910 | 0.250 / 0.267 |
| Screenshot trained on English: English / non-English test | 0.964 / 0.940 | 0.533 / 0.274 |
| Brand ID on Phish-IRIS (15 classes) | accuracy 0.872, macro-F1 0.802 | unseen-brand AUROC 0.828 |

Story in one line: screenshots complement URLs well (fusion is best), but the screenshot model mostly learned
"sparse login page vs busy homepage", a Phish360 collection artifact, and it loses about half its strict-threshold
detection rate under dataset or language shift.

### Waiting on you, with exact steps

1. **Review the PROVISIONAL decisions** in `docs/QUESTIONS.md`: Q4 (own grouped split), Q5 (168 blank screenshots
   excluded), Q6 (folder name as label), Q9 (Phish-IRIS "other" = legitimate; the Phish-IRIS paper describes it as
   legitimate samples), Q10 (Phish-IRIS cropped to 4:3). Each entry says how to change it.
2. **Review and merge the GitHub pull requests** (private repo, one PR per phase, stacked: merge them in order, Phase 0 first).
3. **Deploy the demo (optional):** follow `docs/DEPLOY.md`. The quickest route is Cloudflare Pages: upload the contents of `web/dist`
   after `cd web && npm install && npm run build`. Then put the link in README.md (the "Demo link" line).
4. **Optional demo gallery:** capture screenshots of legitimate pages yourself (a Wikipedia article, your university homepage,
   a news front page, and a real bank or webmail *login* page to show the main weakness). Details are in docs/DEPLOY.md.
5. **Read before the viva:** `docs/learning-log.md` (top to bottom), `docs/viva-cheatsheet.md`, `docs/report.pdf`.

### Things I'd double-check

- **Phish-IRIS copy:** a Kaggle mirror with 3 images fewer than the paper. Results on it also carry a JPEG-format shortcut (trivial AUC 0.995).
- **Small subsets:** per-brand (some brands have 1-4 test images) and per-language (French 85, Spanish 70) numbers have wide CIs.
- **Report title and author line** in `docs/report.md`: adjust to your lab's format, then regenerate the PDF (see below).
- **Full re-run:** `run_all.sh` was assembled from the exact commands used, and spot re-runs of several steps gave byte-identical
  results, but a full from-scratch run (about 4-5 hours) was not repeated end to end.
- **Regenerating the PDF:** it was made by converting Markdown to HTML (npm `marked`) and printing with headless Edge (no pandoc on
  this machine). Simplest alternative: install pandoc, or open the Markdown in VS Code and print it.

## Phase progress

| Phase | State |
|---|---|
| 0 Setup | Done (Python 3.13, CPU only) |
| 1 Data audit | Done for both datasets: grouped splits, leakage and shortcut checks |
| 2 URL baseline | Done: LightGBM val 0.970 (0.943 without artifact features) |
| 3 Image models | Done: fine-tuned MobileNetV3 val 0.967 ± 0.003 |
| 4 Fusion | Done: late fusion val 0.989, TPR@1%FPR 0.81 |
| 5 Final + generalization | Done: see table above and docs/findings-phase5.md |
| 6 Brand ID (optional) | Done |
| 7 Explainability | Done: results/error_analysis.md, results/phase7/ |
| 8 Demo (optional) | Done: web/, bit-exact preprocessing, browser self-tests pass; not deployed (your step) |
| 9 Report | Done: docs/report.md, docs/report.pdf, docs/viva-cheatsheet.md, README.md |

## Git

Local branches, one per phase, each built on the previous one: `main` (Phase 0), then `docs/onedrive-note`,
`feat/phase-1-data-audit`, `feat/phase-2-url-baseline`, `feat/phase-3-image-models`, `feat/phase-4-fusion`,
`feat/phase-5-final-generalization`, `feat/phase-6-brand-id`, `feat/phase-7-explainability`, `feat/phase-8-static-demo`,
and `docs/phase-9-report`. The Phish-IRIS addition is a commit on the Phase 3 branch. Pushed to a private GitHub repo with one PR per branch.
data/, model checkpoints and per-screenshot images are git-ignored. `data/raw/phish360` and `data/raw/phish-iris` are the only copies of the
datasets (the Downloads copies were deleted at your request).
