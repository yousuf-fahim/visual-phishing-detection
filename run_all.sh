#!/usr/bin/env bash
# Reproduce every result in results/ from the raw datasets, in pipeline order.
# Needs: .venv set up from requirements.txt, and the datasets in data/raw/ (see README.md).
# CPU-only time on a 6-core machine: roughly 4-5 hours, mostly image-model training.
# Each step writes its outputs to data/processed/, models/ or results/; logs go to results/*/logs/.
set -euo pipefail
PY=".venv/Scripts/python"; [ -x "$PY" ] || PY=".venv/bin/python"
export PYTHONIOENCODING=utf-8
run() { echo ">>> $*"; "$PY" -W ignore -m "$@"; }

# Phase 0: environment report
run src.check_env

# Phase 1: index, trivial features + perceptual hashes, leakage groups, shortcut check, splits
run src.build_index
run src.image_stats
run src.dedup
run src.shortcut_check
run src.make_splits

# Phase 2: URL baseline
run src.url_features
run src.train_url

# Phase 3: image models
run src.preprocess_images
run src.embed_images
run src.train_frozen
run src.train_image --model cnn --variant full --seed 42
for v in full top; do for s in 42 43 44; do
  run src.train_image --model mobilenetv3 --variant $v --seed $s
done; done
run src.summarize_phase3

# Phase 4: fusion ablation + robustness probe
run src.train_fusion

# Phase 5: one-time test evaluation and generalization
run src.page_language
for s in 42 43 44; do
  run src.train_image --model mobilenetv3 --variant full --seed $s --dataset phish-iris --out results/phase5/image_runs
  run src.train_image --model mobilenetv3 --variant full --seed $s --dataset phish360 --subset english --out results/phase5/image_runs
done
run src.evaluate
run src.generalization

# Phase 6: brand identification (Phish-IRIS)
run src.brand

# Phase 7: explainability and robustness
run src.explain
run src.robustness

# Phase 8: browser demo (model export + JS parity test + static build)
run src.export_onnx
( cd web && npm install --no-audit --no-fund && npm test && npm run build )
echo "All done."
