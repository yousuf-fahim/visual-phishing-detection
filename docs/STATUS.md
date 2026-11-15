# Status

Last updated: 2026-11-15 (autopilot run).

**Phase 1 done for Phish360. Phish-IRIS is still missing**: when it arrives, re-run Phase 1 (parser plus cross-dataset overlap), then continue. 

## Waiting on Fahim

1. **Verify the PROVISIONAL decisions** in `docs/QUESTIONS.md`: Q4 (own grouped split), Q5 (116 blank screenshots excluded), Q6 (folder name is the label), Q9 (Phish-IRIS mapping), Q10 (Phish-IRIS 4:3 crop).

## Phase progress

| Phase | State |
|---|---|
| 0 Setup | Done |
| 1 Data audit | Done for both datasets |
| 2 URL baseline | Done: LightGBM val ROC-AUC 0.970, TPR@1%FPR 0.63 (0.943 / 0.53 without artifact features) |
| 3 Image models | Done: fine-tuned MobileNetV3 val ROC-AUC 0.967 ± 0.003 (3 seeds) |
| 4 Fusion | Done: late fusion val ROC-AUC 0.989, TPR@1%FPR 0.81 (URL 0.63, image 0.57) |
| 5 Final + generalization | Done: test late fusion 0.984; image 0.809 on shortcut-hard pages; cross-dataset 0.91-0.93; non-English 0.940 (see docs/findings-phase5.md) |
| 6 Brand ID (optional) | Done: Phish-IRIS 15-class accuracy 0.872, macro-F1 0.802; unseen-brand AUROC 0.828 |
| 7 Explainability | Done: attention on empty space below forms, not logos; border-only AUC 0.914; robust to JPEG/grayscale, fragile to blur/top-crop |
| 8 Demo (optional) | Done: static site in web/dist, bit-exact JS preprocessing, self-test passes in Edge and Chrome; deploy steps in docs/DEPLOY.md |
| 9 Report | Next |

## Phase 1 headline facts (from results/phase1/)

- Phish360: 10,748 samples, 0 corrupt images, 10 missing HTML files (antivirus, probably).
- Official split leaks (15.5% of its test has a relative in trainval), so we use our own grouped split: 7,451 / 1,591 / 1,590 after excluding 116 blank screenshots.
- Shortcut: trivial image statistics reach ROC-AUC 0.85 (homepage vs login-page confound). This is the floor every image model must beat.

## Git

At the end of the run: create a **private** GitHub repo (Fahim's request), push `main` and every phase branch, and open PRs for review. data/, screenshots and models/ are never pushed. The Phish360 copies in Downloads were deleted at Fahim's request: `data/raw/phish360` is now the only copy.


No remote. Branches are local, one per phase, each built on the previous one: `main` (Phase 0), then `docs/onedrive-note`, then `feat/phase-1-data-audit`.
