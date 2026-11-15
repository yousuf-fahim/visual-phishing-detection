# Status

Last updated: 2026-11-15 (autopilot run).

**Phase 1 done for Phish360. Phish-IRIS is still missing**: when it arrives, re-run Phase 1 (parser plus cross-dataset overlap), then continue. Moving on to Phase 2 with Phish360.

## Waiting on Fahim

1. **Phish-IRIS:** download it and tell me the path (or put it in `data/raw/phish-iris/`).
2. **Verify the PROVISIONAL decisions** in `docs/QUESTIONS.md`: Q4 (own grouped split), Q5 (116 blank screenshots excluded), Q6 (folder name is the label).

## Phase progress

| Phase | State |
|---|---|
| 0 Setup | Done |
| 1 Data audit | Done for Phish360 (branch `feat/phase-1-data-audit`); Phish-IRIS parts pending |
| 2 URL baseline | Next |
| 3-9 | Not started |

## Phase 1 headline facts (from results/phase1/)

- Phish360: 10,748 samples, 0 corrupt images, 10 missing HTML files (antivirus, probably).
- Official split leaks (15.5% of its test has a relative in trainval), so we use our own grouped split: 7,451 / 1,591 / 1,590 after excluding 116 blank screenshots.
- Shortcut: trivial image statistics reach ROC-AUC 0.85 (homepage vs login-page confound). This is the floor every image model must beat.

## Git

No remote. Branches are local, one per phase, each built on the previous one: `main` (Phase 0), then `docs/onedrive-note`, then `feat/phase-1-data-audit`.
