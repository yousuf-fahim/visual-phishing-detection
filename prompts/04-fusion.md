# Phase 4: Fusion and ablation (60 min)

Run after /clear. Paste everything below the line.

Checkpoint: when does fusing two modalities help, and what would it mean if fusion did not beat the best single model?

---

Phase 4: fuse the screenshot and URL features.

1. Extract a fixed embedding for every screenshot from the best image model (frozen), and save it.
2. Build two fusion models: (a) early fusion, concatenating the image embedding with the URL features and training a small MLP; (b) late fusion, combining the probabilities from the separate URL and image models.
3. Produce the ablation table on the validation set: URL only, image only, fused (early), fused (late). Use 3 seeds and report mean and standard deviation for precision, recall, F1, ROC-AUC and TPR at 1% FPR.
4. Add a robustness probe: set the URL features to missing (zeros) at inference, and then the image to a blank canvas, and show how each fused model degrades. Explain what this tells us.
5. State plainly whether fusion helps. Do not overstate: Phish360's authors have already published multimodal work, so this is a reproduction-scale ablation, not a novelty claim.
6. Write it as src/train_fusion.py with fixed seeds, save results to results/, append the Phase 4 entry to docs/learning-log.md and commit.
