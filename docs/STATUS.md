# Status

**WAITING at the start of Phase 1: datasets are downloading. data/raw/ is checked every 15 minutes, and Phase 1 starts automatically once the download is stable.**
Last updated: 2026-11-15 (autopilot run).

## What you need to do (then restart autopilot)

1. Download **Phish360** from https://web.cs.hacettepe.edu.tr/~selman/phish360-dataset/
   and extract it into `data/raw/phish360/`.
2. Download **Phish-IRIS** from https://web.cs.hacettepe.edu.tr/~selman/phish-iris-dataset/
   (or Mendeley Data, DOI 10.17632/tk4fkswtj5.1) and extract it into `data/raw/phish-iris/`.
   The folder names are a suggestion: Phase 1 inspects whatever layout it finds.
3. Read `docs/QUESTIONS.md` (decisions made without you, all easy to reverse).
4. Run `claude` in this folder and paste: `Read AUTOPILOT.md and follow it exactly. Continue from docs/STATUS.md (Phase 1).`

## Phase progress

| Phase | State |
|---|---|
| 0 Setup | Done. Committed. |
| 1 Data audit | Not started: blocked, no datasets in data/raw/ |
| 2-9 | Not started |

## Phase 0 summary

- `.venv` with Python 3.13.3 (the only Python on this machine, see QUESTIONS.md Q1).
- `requirements.txt` with pinned versions. All 19 libraries installed and import cleanly.
- No GPU: torch 2.15.0+cpu, 6 CPU threads. Later phases should prefer frozen-backbone
  embeddings and small images to keep training time down.
- tldextract runs offline (`src/domain_utils.py`). Checked: `example.co.uk` extracted with no network.
- Environment report: `results/env.json` (written by `src/check_env.py`).
- Git repo initialised, first commit on `main`.

## Next

Phase 1 (`prompts/01-data-audit.md`), as soon as both datasets are in data/raw/.
