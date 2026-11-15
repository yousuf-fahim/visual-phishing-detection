# Phase 0: Setup (20-30 min)

Run first. Open a terminal in this folder, run `claude`, then paste everything below the line.

You should see: a working virtual environment, requirements.txt, library versions printed, and one commit.
Checkpoint: which Python version did it pick, and is a GPU available?

---

I'm Fahim, a final-year CSE student building a visual phishing detection study for my Neural Network lab. I'm comfortable with web development and have done an ML lab on phishing URL detection, but I'm new to computer vision, so teach me as we go.

This folder already contains CLAUDE.md, PLAN.md, AUTOPILOT.md, README.md and prompts/. Read CLAUDE.md and PLAN.md first and follow the rules in CLAUDE.md for the whole project, especially the safety and ethics rules. Then set up the environment:

1. Create a Python virtual environment (.venv, Python 3.10-3.12) and a requirements.txt with: numpy, pandas, scikit-learn, lightgbm, matplotlib, seaborn, opencv-python, pillow, torch, torchvision, timm, grad-cam, onnx, onnxruntime, tqdm, imagehash, tldextract, langdetect, beautifulsoup4. Install them. If something fails to install, tell me what failed and propose a fix instead of silently swapping libraries. Configure tldextract to work offline.
2. Confirm the folders data/raw, data/processed, src, notebooks, models, results, docs and web exist, and that .gitignore is in place.
3. Print library versions and check whether a GPU is available.
4. If this is not a git repository yet, run git init. Make the first commit.
5. Tell me in a few lines what you set up and what the next step is. Do not start Phase 1.
