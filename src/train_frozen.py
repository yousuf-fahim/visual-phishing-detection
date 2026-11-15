"""Phase 3: train small heads on frozen-backbone embeddings (from src.embed_images).

Run: .venv/Scripts/python -m src.train_frozen

For every backbone (resnet18, deit_tiny, mobilenetv3) and variant (full, top):
  * logistic-regression head (standardised features, L2 C=0.1, fixed in advance), deterministic
  * MLP head (one hidden layer of 256), seeds 42/43/44, early stopping on val loss
Validation set only. Writes results in the same format as src.train_image
(results/phase3/runs/*.json and preds/*.csv), so src.summarize_phase3 can compare them all.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.metrics import binary_metrics
from src.train_url import set_seed

OUT = Path("results/phase3")
BACKBONE_PARAMS = {"resnet18": 11.18e6, "deit_tiny": 5.52e6, "mobilenetv3": 1.52e6}  # frozen, from timm


def mlp_head(Xtr, ytr, Xva, yva, seed, max_epochs=200, patience=15):
    set_seed(seed)
    scaler = StandardScaler().fit(Xtr)
    Xtr_t = torch.tensor(scaler.transform(Xtr), dtype=torch.float32)
    Xva_t = torch.tensor(scaler.transform(Xva), dtype=torch.float32)
    ytr_t, yva_t = torch.tensor(ytr, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    model = nn.Sequential(nn.Dropout(0.2), nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for b in torch.randperm(len(Xtr_t), generator=gen).split(128):
            opt.zero_grad()
            loss_fn(model(Xtr_t[b]).squeeze(1), ytr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva_t).squeeze(1), yva_t).item()
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        return torch.sigmoid(model(Xva_t).squeeze(1)).numpy(), sum(p.numel() for p in model.parameters())


def main() -> None:
    for sub in ("runs", "preds"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv("data/processed/splits.csv", keep_default_na=False)
    use = (splits.dataset == "phish360") & (splits.exclude == "")
    tr = np.flatnonzero(use & (splits.split == "train"))
    va = np.flatnonzero(use & (splits.split == "val"))
    y = splits.label_binary.to_numpy()

    for backbone in BACKBONE_PARAMS:
        for variant in ("full", "top"):
            emb = np.load(f"data/processed/emb_{backbone}_{variant}.npy")
            Xtr, Xva = emb[tr], emb[va]
            jobs = [("lr", 42)] + [("mlp", s) for s in (42, 43, 44)]
            for head, seed in jobs:
                t0 = time.time()
                if head == "lr":
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
                    clf.fit(Xtr, y[tr])
                    scores, head_params = clf.predict_proba(Xva)[:, 1], Xtr.shape[1] + 1
                else:
                    scores, head_params = mlp_head(Xtr, y[tr], Xva, y[va], seed)
                run = f"frozen-{backbone}-{head}_{variant}_s{seed}"
                pd.DataFrame({"path": splits.path.to_numpy()[va], "label_binary": y[va], "score": scores}
                             ).to_csv(OUT / "preds" / f"{run}.csv", index=False)
                result = {"run": run, "model": f"frozen-{backbone}-{head}", "training": "frozen backbone + head",
                          "variant": variant, "seed": seed, "params": int(BACKBONE_PARAMS[backbone] + head_params),
                          "trainable_params": int(head_params), "train_seconds": round(time.time() - t0),
                          **binary_metrics(y[va], scores)}
                (OUT / "runs" / f"{run}.json").write_text(json.dumps(result, indent=2))
                print(f"{run:40s} AUC={result['roc_auc']:.4f} F1={result['f1']:.4f} "
                      f"TPR@1%FPR={result['tpr_at_1pct_fpr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
