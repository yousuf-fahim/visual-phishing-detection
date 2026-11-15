"""Phase 6 (stretch): which brand does a phishing page imitate? Phish-IRIS only.

Run: .venv/Scripts/python -m src.brand            (all runs, then summary)
     .venv/Scripts/python -m src.brand --summary  (summary only, from saved runs)

1. Closed set: 15 classes (14 brands + 'other'), fine-tuned MobileNetV3-small (Phase 3's best
   backbone), seeds 42/43/44, Phish-IRIS grouped splits. Class-weighted cross-entropy, because
   'other' is about half of the data. Early stopping on validation macro-F1. Reported on the
   Phish-IRIS TEST split: accuracy, macro-F1, confusion matrix.
2. Open set: 3 brands, drawn at random with seed 42 BEFORE any result was seen, are removed from
   training entirely. The model learns the other 12 classes. At test time, its top softmax
   probability is the "I know this" confidence. AUROC measures how well low confidence
   separates the unseen brands from known-class test pages.
3. Brand-vs-domain consistency check: skipped. Phish-IRIS has no URLs, so there is no domain to
   compare the predicted brand against.

Outputs: results/phase6/{closed_set,open_set}_runs.csv, summary.json, confusion_matrix.png
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn

from src.train_image import EVAL_TF, TRAIN_TF

SEEDS = [42, 43, 44]
OUT = Path("results/phase6")
EPOCHS, PATIENCE = 15, 4


def held_out_brands(brands: list[str]) -> list[str]:
    """Three brands chosen at random (seed 42) from the sorted brand list."""
    return sorted(random.Random(42).sample(sorted(brands), 3))


class BrandSet(torch.utils.data.Dataset):
    def __init__(self, images, rows, labels, tf):
        self.images, self.rows, self.labels, self.tf = images, rows, labels, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = torch.from_numpy(np.array(self.images[self.rows[i]])).permute(2, 0, 1)
        return self.tf(img), int(self.labels[i])


@torch.no_grad()
def predict_proba(model, images, rows) -> np.ndarray:
    model.eval()
    loader = torch.utils.data.DataLoader(BrandSet(images, rows, np.zeros(len(rows)), EVAL_TF), batch_size=128)
    return np.concatenate([torch.softmax(model(x), 1).numpy() for x, _ in loader])


def train(images, tr_rows, tr_y, va_rows, va_y, n_classes, seed) -> tuple[nn.Module, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = timm.create_model("mobilenetv3_small_100.lamb_in1k", pretrained=True, num_classes=n_classes)
    counts = np.bincount(tr_y, minlength=n_classes).astype(float)
    weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)  # rare brands count as much as 'other'
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loader = torch.utils.data.DataLoader(BrandSet(images, tr_rows, tr_y, TRAIN_TF), batch_size=32, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))
    best_f1, best_state, best_epoch, t0 = -1.0, None, 0, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        sched.step()
        f1 = f1_score(va_y, predict_proba(model, images, va_rows).argmax(1), average="macro")
        print(f"    epoch {epoch:2d} val macro-F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "val_macro_f1": round(best_f1, 4), "train_seconds": round(time.time() - t0)}


def load():
    idx = pd.read_csv("data/processed/index.csv", keep_default_na=False)
    splits = pd.read_csv("data/processed/splits.csv", keep_default_na=False)
    assert (idx.path == splits.path).all()
    d = splits.assign(folder_class=idx.folder_class)
    d = d[(d.dataset == "phish-iris") & (d.exclude == "")]
    return d, np.load("data/processed/img224_full.npy", mmap_mode="r")


def run_all() -> None:
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    d, images = load()
    classes = sorted(d.folder_class.unique())  # 14 brands + 'other'
    brands = [c for c in classes if c != "other"]
    unknown = held_out_brands(brands)
    rows_of = {s: d[d.split == s] for s in ("train", "val", "test")}

    closed, open_ = [], []
    for seed in SEEDS:
        # ---- closed set: all 15 classes ----
        cid = {c: i for i, c in enumerate(classes)}
        tr, va, te = (rows_of[s] for s in ("train", "val", "test"))
        print(f"closed set, seed {seed}", flush=True)
        model, info = train(images, tr.index.to_numpy(), tr.folder_class.map(cid).to_numpy(),
                            va.index.to_numpy(), va.folder_class.map(cid).to_numpy(), len(classes), seed)
        p = predict_proba(model, images, te.index.to_numpy())
        y_true, y_pred = te.folder_class.map(cid).to_numpy(), p.argmax(1)
        is_brand = (te.folder_class != "other").to_numpy()  # accuracy on phishing pages only
        closed.append({"seed": seed, **info, "test_accuracy": accuracy_score(y_true, y_pred),
                       "test_macro_f1": f1_score(y_true, y_pred, average="macro"),
                       "test_brand_only_accuracy": accuracy_score(y_true[is_brand], y_pred[is_brand])})
        np.save(OUT / f"closed_test_proba_s{seed}.npy", p)

        # ---- open set: train without the 3 held-out brands ----
        known = [c for c in classes if c not in unknown]
        kid = {c: i for i, c in enumerate(known)}
        trk, vak = tr[tr.folder_class.isin(known)], va[va.folder_class.isin(known)]
        print(f"open set (unknown = {unknown}), seed {seed}", flush=True)
        model, info = train(images, trk.index.to_numpy(), trk.folder_class.map(kid).to_numpy(),
                            vak.index.to_numpy(), vak.folder_class.map(kid).to_numpy(), len(known), seed)
        te_known = te[te.folder_class.isin(known)]
        # unseen brands: every sample from any split was excluded from training, so all are usable
        unk = d[d.folder_class.isin(unknown)]
        pk, pu = predict_proba(model, images, te_known.index.to_numpy()), predict_proba(model, images, unk.index.to_numpy())
        conf = np.concatenate([pk.max(1), pu.max(1)])
        is_unknown = np.concatenate([np.zeros(len(pk)), np.ones(len(pu))])
        open_.append({"seed": seed, **info, "unknown_brands": ",".join(unknown), "n_known_test": len(pk),
                      "n_unknown": len(pu),
                      # AUROC of (1 - max softmax) for flagging unknown brands; 0.5 = chance
                      "unknown_auroc": roc_auc_score(is_unknown, 1 - conf),
                      "known_test_accuracy": accuracy_score(te_known.folder_class.map(kid), pk.argmax(1)),
                      "mean_conf_known": pk.max(1).mean(), "mean_conf_unknown": pu.max(1).mean()})
        pd.DataFrame(closed).to_csv(OUT / "closed_set_runs.csv", index=False)
        pd.DataFrame(open_).to_csv(OUT / "open_set_runs.csv", index=False)


def summary() -> None:
    d, _ = load()
    classes = sorted(d.folder_class.unique())
    te = d[d.split == "test"]
    closed, open_ = pd.read_csv(OUT / "closed_set_runs.csv"), pd.read_csv(OUT / "open_set_runs.csv")
    agg = lambda df, cols: {c: {"mean": round(df[c].mean(), 4), "std": round(df[c].std(), 4)} for c in cols}
    s = {"closed_set": {"n_classes": len(classes), "n_test": len(te), "seeds": len(closed),
                        **agg(closed, ["test_accuracy", "test_macro_f1", "test_brand_only_accuracy"])},
         "open_set": {"unknown_brands": open_.unknown_brands.iloc[0], "n_unknown": int(open_.n_unknown.iloc[0]),
                      "n_known_test": int(open_.n_known_test.iloc[0]),
                      **agg(open_, ["unknown_auroc", "known_test_accuracy", "mean_conf_known", "mean_conf_unknown"])},
         "brand_domain_consistency": "skipped: Phish-IRIS has no URLs"}
    (OUT / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))

    # Confusion matrix summed over the 3 seeds, row-normalised (each row = one true class)
    cid = {c: i for i, c in enumerate(classes)}
    y_true = te.folder_class.map(cid).to_numpy()
    cm = sum(confusion_matrix(y_true, np.load(OUT / f"closed_test_proba_s{s}.npy").argmax(1), labels=range(len(classes)))
             for s in SEEDS)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right")
    ax.set_yticks(range(len(classes)), [f"{c} (n={int(n)})" for c, n in zip(classes, cm.sum(1) // len(SEEDS))])
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Phish-IRIS brand identification (test, 3 seeds pooled, row-normalised)")
    fig.colorbar(im, fraction=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    if not ap.parse_args().summary:
        run_all()
    summary()
