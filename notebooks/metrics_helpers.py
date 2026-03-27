"""
Proposal-aligned multi-label metrics (micro/macro F1, per-label PR + ROC-AUC, confusion).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def multilabel_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_names: tuple[str, ...] | list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    ``y_true``, ``y_pred``: (N, L) binary. ``y_prob``: (N, L) in [0, 1] or logits passed through sigmoid before calling.
    Returns (per-label DataFrame, summary dict with f1_micro, f1_macro).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    rows: list[dict[str, float | str]] = []
    for i, name in enumerate(label_names):
        try:
            auc = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
        except ValueError:
            auc = float("nan")
        rows.append(
            {
                "label": name,
                "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
                "roc_auc": auc,
            }
        )
    per_label = pd.DataFrame(rows)
    summary = {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return per_label, summary


def per_label_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: tuple[str, ...] | list[str],
) -> dict[str, np.ndarray]:
    """2×2 confusion matrix per label: [[TN, FP], [FN, TP]] for binary toxicity."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(label_names):
        out[name] = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
    return out


def torch_parameter_count(model: object) -> int:
    """Trainable + non-trainable parameters (HF models: use ``model.num_parameters()`` if available)."""
    if hasattr(model, "num_parameters"):
        return int(model.num_parameters())
    params = getattr(model, "parameters", None)
    if callable(params):
        return sum(p.numel() for p in params())
    raise TypeError("Expected a PyTorch module or HF pretrained model")
