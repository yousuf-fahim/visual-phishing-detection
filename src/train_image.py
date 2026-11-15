"""Phase 3: train a screenshot classifier end to end (all weights updated).

Run, e.g.:
  .venv/Scripts/python -m src.train_image --model cnn --variant full --seed 42
  .venv/Scripts/python -m src.train_image --model mobilenetv3 --variant top --seed 43

Models
  cnn          small CNN from scratch (4 conv blocks)
  mobilenetv3  MobileNetV3-small pretrained on ImageNet, all layers fine-tuned

Augmentation is deliberately light, and there are NO horizontal flips: in a
screenshot, layout carries meaning (logo top-left, text reads left to right), and
a mirrored login page is not a page anyone would see.

Early stopping: after each epoch we score the validation set. We keep the epoch
with the best validation ROC-AUC and stop once it hasn't improved for PATIENCE epochs.
The test set is never touched here.

Outputs (run name = {model}_{variant}_s{seed})
  models/img_{run}.pt                      best checkpoint
  results/phase3/runs/{run}.json           metrics at best epoch, params, training time
  results/phase3/history/{run}.csv         per-epoch train loss, val loss, val AUC
  results/phase3/preds/{run}.csv           per-sample validation scores
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torchvision.transforms import v2

from src.metrics import binary_metrics

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PATIENCE = 4
OUT = Path("results/phase3")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------- data ----------------

TRAIN_TF = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=True),  # small crops
    v2.RandomApply([v2.JPEG(quality=(50, 95))], p=0.3),        # re-compression artefacts
    v2.ToDtype(torch.float32, scale=True),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
    v2.Normalize(MEAN, STD),
])
EVAL_TF = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])


class ScreenshotSet(torch.utils.data.Dataset):
    def __init__(self, images: np.ndarray, rows: np.ndarray, labels: np.ndarray, transform):
        self.images, self.rows, self.labels, self.tf = images, rows, labels, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = torch.from_numpy(np.array(self.images[self.rows[i]])).permute(2, 0, 1)  # uint8 (3,224,224)
        return self.tf(img), torch.tensor(self.labels[i], dtype=torch.float32)


def load_split(variant: str, dataset: str = "phish360", subset: str = "all"):
    """Train/val rows for one dataset. subset='english' keeps only pages whose
    detected language is English (Phase 5 cross-language test)."""
    idx = pd.read_csv("data/processed/index.csv", keep_default_na=False)
    splits = pd.read_csv("data/processed/splits.csv", keep_default_na=False)
    assert (idx.path == splits.path).all()
    images = np.load(f"data/processed/img224_{variant}.npy", mmap_mode="r")
    use = (splits.dataset == dataset) & (splits.exclude == "")
    if subset == "english":
        lang = pd.read_csv("data/processed/languages.csv", keep_default_na=False)
        assert (lang.path == splits.path).all()
        use &= lang.language == "en"
    out = {}
    for s in ("train", "val"):
        rows = np.flatnonzero(use & (splits.split == s))
        out[s] = (rows, splits.label_binary.to_numpy()[rows], splits.path.to_numpy()[rows])
    return images, out


# ---------------- models ----------------

class SmallCNN(nn.Module):
    """Four conv blocks (conv-BN-ReLU-maxpool), channel width doubling each block,
    then global average pooling and one linear output. About 0.4M parameters."""

    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        layers, c_in = [], 3
        for i, c in enumerate(widths):
            stride = 2 if i == 0 else 1  # the first conv also halves 224 -> 112
            layers += [nn.Conv2d(c_in, c, 3, stride=stride, padding=1, bias=False),
                       nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c_in, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class TimmBinary(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.net = timm.create_model(name, pretrained=True, num_classes=1)

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_model(name: str):
    if name == "cnn":
        return SmallCNN(), 1e-3
    if name == "mobilenetv3":
        return TimmBinary("mobilenetv3_small_100.lamb_in1k"), 5e-4
    raise ValueError(name)


# ---------------- training ----------------

@torch.no_grad()
def predict(model, loader, loss_fn):
    model.eval()
    scores, losses, n = [], 0.0, 0
    for x, y in loader:
        logits = model(x)
        losses += loss_fn(logits, y).item() * len(y)
        n += len(y)
        scores.append(torch.sigmoid(logits).numpy())
    return np.concatenate(scores), losses / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cnn", "mobilenetv3"])
    ap.add_argument("--variant", default="full", choices=["full", "top"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dataset", default="phish360", choices=["phish360", "phish-iris"])
    ap.add_argument("--subset", default="all", choices=["all", "english"])
    ap.add_argument("--out", default="results/phase3", help="folder for runs/, history/, preds/")
    args = ap.parse_args()
    global OUT
    OUT = Path(args.out)

    torch.set_num_threads(6)
    set_seed(args.seed)
    run = f"{args.model}_{args.variant}_s{args.seed}"
    if args.dataset != "phish360" or args.subset != "all":  # Phase 3 run names stay unchanged
        run = f"{args.dataset}-{args.subset}-{run}"
    for sub in ("runs", "history", "preds"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    images, data = load_split(args.variant, args.dataset, args.subset)
    tr_rows, tr_y, _ = data["train"]
    va_rows, va_y, va_paths = data["val"]
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(ScreenshotSet(images, tr_rows, tr_y, TRAIN_TF),
                                               batch_size=args.batch, shuffle=True, generator=gen)
    val_loader = torch.utils.data.DataLoader(ScreenshotSet(images, va_rows, va_y, EVAL_TF), batch_size=128)

    model, lr = build_model(args.model)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    print(f"{run}: {n_params / 1e6:.2f}M params, train={len(tr_rows)} val={len(va_rows)}", flush=True)

    history, best_auc, best_epoch, best_scores = [], -1.0, 0, None
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(y)
            n += len(y)
        sched.step()
        scores, val_loss = predict(model, val_loader, loss_fn)
        m = binary_metrics(va_y, scores)
        history.append({"epoch": epoch, "train_loss": run_loss / n, "val_loss": val_loss,
                        "val_auc": m["roc_auc"], "val_tpr_at_1pct_fpr": m["tpr_at_1pct_fpr"],
                        "epoch_seconds": time.time() - t0})
        print(f"  epoch {epoch:2d}  train_loss={run_loss / n:.4f}  val_loss={val_loss:.4f}  "
              f"val_auc={m['roc_auc']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if m["roc_auc"] > best_auc:
            best_auc, best_epoch, best_scores = m["roc_auc"], epoch, scores
            torch.save(model.state_dict(), f"models/img_{run}.pt")
        elif epoch - best_epoch >= PATIENCE:
            print(f"  early stop: no val AUC gain for {PATIENCE} epochs", flush=True)
            break

    train_seconds = time.time() - t_start
    pd.DataFrame(history).to_csv(OUT / "history" / f"{run}.csv", index=False)
    pd.DataFrame({"path": va_paths, "label_binary": va_y, "score": best_scores}).to_csv(
        OUT / "preds" / f"{run}.csv", index=False)
    result = {"run": run, "dataset": args.dataset, "subset": args.subset, "model": args.model, "training": "fine-tuned" if args.model != "cnn" else "from scratch",
              "variant": args.variant, "seed": args.seed, "params": n_params, "best_epoch": best_epoch,
              "epochs_run": len(history), "train_seconds": round(train_seconds),
              **binary_metrics(va_y, best_scores)}
    (OUT / "runs" / f"{run}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
