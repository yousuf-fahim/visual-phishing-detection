"""Phase 3: frozen-backbone embeddings for every screenshot.

Run: .venv/Scripts/python -m src.embed_images

A pretrained network (trained on ImageNet's 1.2M everyday photos) is used as a
fixed feature extractor: we take the vector just before its final classification
layer (its "embedding"), and later train only a small head on top. That is far
cheaper on a CPU than fine-tuning all the weights, and the vectors are reused for
fusion in Phase 4.

Output: data/processed/emb_{backbone}_{variant}.npy  float32 (N, D), rows in index.csv order.
"""
import time
from pathlib import Path

import numpy as np
import timm
import torch

BACKBONES = {  # timm name -> short name
    "resnet18.a1_in1k": "resnet18",
    "deit_tiny_patch16_224.fb_in1k": "deit_tiny",
    "mobilenetv3_small_100.lamb_in1k": "mobilenetv3",
}
VARIANTS = ["full", "top"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_input(batch_uint8: np.ndarray) -> torch.Tensor:
    """(B,H,W,3) uint8 -> normalised (B,3,H,W) float tensor."""
    x = torch.from_numpy(np.ascontiguousarray(batch_uint8)).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN) / STD


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(6)
    for variant in VARIANTS:
        imgs = np.load(f"data/processed/img224_{variant}.npy", mmap_mode="r")
        for timm_name, short in BACKBONES.items():
            out = Path(f"data/processed/emb_{short}_{variant}.npy")
            if out.exists():
                print(f"skip {out} (exists)")
                continue
            model = timm.create_model(timm_name, pretrained=True, num_classes=0).eval()  # no classifier head
            t0, chunks = time.time(), []
            for start in range(0, len(imgs), 64):
                chunks.append(model(to_input(imgs[start:start + 64])).numpy())
            emb = np.concatenate(chunks).astype(np.float32)
            np.save(out, emb)
            print(f"{short:12s} {variant:4s} dim={emb.shape[1]:5d}  {time.time() - t0:6.0f}s  -> {out}")


if __name__ == "__main__":
    main()
