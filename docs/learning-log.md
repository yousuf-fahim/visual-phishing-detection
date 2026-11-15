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
