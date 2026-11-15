# Phase 3: Image models (90-120 min)

Run after /clear. Paste everything below the line.

Checkpoint: why did we avoid horizontal flips as augmentation for screenshots? Why does a pretrained model help with a small dataset?

---

Phase 3: screenshot classifiers on Phish360, using the grouped splits.

1. Preprocessing: resize every screenshot onto a fixed 224x224 canvas (keep aspect ratio and pad), so that image size is not a shortcut. Also build a variant using only the top part of the page ("above the fold"). Explain why that region might matter.
2. Augmentation: light only (small crops, color jitter, blur, JPEG compression). Do not use horizontal flips, and explain why: layout carries meaning in screenshots.
3. Models: (a) a small CNN from scratch, (b) pretrained ResNet18 or MobileNetV3-small, fine-tuned, (c) optionally a small ViT from timm if the weights are available. Detect CPU vs GPU and estimate training time before starting. If it will take longer than about 40 minutes per model, propose faster options (frozen backbone plus a trained head, smaller images, fewer epochs) and use them.
4. Compare full-page vs above-the-fold inputs for the best model.
5. Train with early stopping on validation. Run the best configuration with 3 seeds and report mean and standard deviation. Validation set only.
6. Plot learning curves, report precision, recall, F1, ROC-AUC and TPR at 1% FPR, and print parameter counts and training time.
7. Write everything as scripts in src/ with fixed seeds. Save checkpoints to models/. Append the Phase 3 entry to docs/learning-log.md and commit.
