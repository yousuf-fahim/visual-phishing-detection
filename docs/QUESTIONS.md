# Decisions made while Fahim was away

Each entry: what came up, the options, what I chose, and how to change it.

## Q1 (Phase 0): Python version

- **What came up:** the plan asks for Python 3.10-3.12, but this machine only has 3.13.3.
- **Options:** (a) use 3.13; (b) install 3.12 system-wide. Option (b) changes files outside the project, which AUTOPILOT counts as a blocker.
- **Chose:** (a). Every required library has 3.13 wheels and passed an import check (`results/env.json`).
- **To change:** install Python 3.12 from python.org, delete `.venv`, then run
  `py -3.12 -m venv .venv` and `.venv/Scripts/python -m pip install -r requirements.txt`.

## Q2 (Phase 0): git branches

- **What came up:** your global rules say never commit to `main` and always open a PR. The project CLAUDE.md says to commit at the end of each phase, and Phase 0 says to run `git init`. A new repo has no remote, so no PR is possible, and the first commit has to go on `main`.
- **Chose:** the initial commit goes on `main`. Each later phase gets its own branch (`feat/phase-1-data-audit`, and so on), committed locally. Nothing is pushed or merged, since there's no remote and creating a GitHub repo is your call.
- **To change:** create the GitHub repo, `git remote add origin <url>`, `git push -u origin main`, then push each phase branch and open PRs.

## Q3 (Phase 0): the project lives in OneDrive

- **What came up:** the folder is under OneDrive, so `.venv` (thousands of files, ~1 GB) and later the datasets and models will be synced. That can slow OneDrive down or lock files while it syncs.
- **Chose:** left everything where it is, because moving it would change files outside the project.
- **To change:** right-click `.venv` and `data/` in Explorer and choose "Free up space" / exclude them from sync, or move the project to a non-synced folder such as `C:\dev\phishing-kit`.

## Note: OpenCV installed twice

`grad-cam` pulls in `opencv-python-headless` alongside `opencv-python`. Both installed and `import cv2` works. If cv2 ever breaks, uninstall both and reinstall just `opencv-python`.
