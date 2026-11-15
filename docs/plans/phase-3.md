# Phase 3 plan: image models (CPU only)

Measured on this machine (Ryzen 5 7500F, 6 cores, no GPU), batch 32 at 224x224:
MobileNetV3-small trains at ~114 img/s (1.1 min/epoch), ResNet18 at ~31 img/s (4 min/epoch), DeiT-tiny at ~33 img/s (3.7 min/epoch).

1. `src/preprocess_images.py`: decode each screenshot once and letterbox it onto a 224x224 canvas
   (keep aspect, pad with ImageNet-mean grey, which is zero after normalisation). Two variants:
   `full` (the whole 1280x960 viewport) and `top` (top half, 1280x480 "above the fold").
   Stored as uint8 .npy memmaps in data/processed (git-ignored), aligned with index.csv rows.
2. `src/train_image.py`: fine-tuning with light augmentation (crop scale 0.85-1, colour jitter, blur, JPEG),
   no flips. AdamW, cosine LR, early stopping on val ROC-AUC (patience 4, max 15 epochs).
   - (a) small CNN from scratch (4 conv blocks, about 0.4M params)
   - (b) MobileNetV3-small, ImageNet-pretrained, fully fine-tuned
3. `src/embed_images.py` + `src/train_frozen.py`: ResNet18, DeiT-tiny and MobileNetV3-small as frozen
   feature extractors (fine-tuning would take over 40 min per model), with a logistic-regression head. 3 seeds.
   The embeddings are reused for fusion in Phase 4.
4. Pick the best configuration by val ROC-AUC, run 3 seeds, and compare full vs top for it.
5. Learning curves, metrics table, parameter counts and training time, all in results/phase3/.
   Every model is also compared with the Phase 1 trivial-feature floor (ROC-AUC 0.85).
