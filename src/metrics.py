"""Shared evaluation metrics, used by every phase from Phase 2 on.

We never report accuracy alone. Wrongly flagging a legitimate site is costly,
so besides ROC-AUC we report the true-positive rate (share of phishing caught)
at a 1% false-positive rate: "if we only allow 1 in 100 legit pages to be
flagged, how much phishing do we still catch?"
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def tpr_at_fpr(y_true, score, max_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(tpr[fpr <= max_fpr].max())


def binary_metrics(y_true, score, threshold: float = 0.5) -> dict:
    """score = predicted probability of phishing (label 1)."""
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    pred = (score >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score),
        "tpr_at_1pct_fpr": tpr_at_fpr(y_true, score),
    }
