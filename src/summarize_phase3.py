"""Phase 3: collect every image-model run into tables and plot learning curves.

Run: .venv/Scripts/python -m src.summarize_phase3

Outputs
  results/phase3/metrics_runs.csv      one row per run
  results/phase3/metrics_summary.csv   mean/std over seeds per (model, variant)
  results/phase3/learning_curves.png   train/val loss and val AUC per epoch (fine-tuned models)
  results/phase3/model_comparison.png  val ROC-AUC and TPR@1%FPR, with the Phase 1 trivial-feature floor
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("results/phase3")
METRICS = ["precision", "recall", "f1", "roc_auc", "tpr_at_1pct_fpr"]


def main() -> None:
    runs = pd.DataFrame([json.loads(p.read_text()) for p in sorted((OUT / "runs").glob("*.json"))])
    runs.to_csv(OUT / "metrics_runs.csv", index=False)

    g = runs.groupby(["model", "training", "variant"])
    summary = g[METRICS].agg(["mean", "std"]).round(4)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary["n_seeds"] = g.size()
    summary["params"] = g.params.first()
    summary["train_seconds_mean"] = g.train_seconds.mean().round()
    summary = summary.reset_index().sort_values("roc_auc_mean", ascending=False)
    summary.to_csv(OUT / "metrics_summary.csv", index=False)
    print(summary[["model", "variant", "n_seeds", "roc_auc_mean", "roc_auc_std", "tpr_at_1pct_fpr_mean",
                   "tpr_at_1pct_fpr_std", "f1_mean", "params", "train_seconds_mean"]].to_string(index=False))

    # Learning curves for end-to-end trained models
    hist_files = sorted((OUT / "history").glob("*.csv"))
    if hist_files:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for f in hist_files:
            h = pd.read_csv(f)
            line, = axes[0].plot(h.epoch, h.train_loss, label=f"{f.stem} train")
            axes[0].plot(h.epoch, h.val_loss, "--", color=line.get_color(), label=f"{f.stem} val")
            axes[1].plot(h.epoch, h.val_auc, color=line.get_color(), label=f.stem)
        axes[0].set_title("Loss (solid = train, dashed = val)")
        axes[0].set_xlabel("epoch")
        axes[1].set_title("Validation ROC-AUC")
        axes[1].set_xlabel("epoch")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "learning_curves.png", dpi=120)
        plt.close(fig)

    # Comparison chart with the trivial-feature floor from Phase 1
    floor = json.loads(Path("results/phase1/shortcut_check.json").read_text())["phish360"]["lightgbm"]
    s = summary.sort_values("roc_auc_mean")
    labels = s.model + " / " + s.variant
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.35 * len(s) + 1.5))
    for ax, m, fl in [(axes[0], "roc_auc", floor["roc_auc"]), (axes[1], "tpr_at_1pct_fpr", floor["tpr_at_1pct_fpr"])]:
        ax.barh(labels, s[f"{m}_mean"], xerr=s[f"{m}_std"].fillna(0), color="#4C72B0")
        ax.axvline(fl, color="#C44E52", ls="--", label=f"trivial features only ({fl:.2f})")
        ax.set_title(f"validation {m}")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_xlim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=120)


if __name__ == "__main__":
    main()
