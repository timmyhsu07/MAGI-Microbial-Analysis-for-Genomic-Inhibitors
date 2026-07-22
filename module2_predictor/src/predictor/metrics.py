"""Metric calculation with explicit undefined cases.

Per-cluster slices are often tiny, so some metrics are mathematically
undefined. The report records those as JSON nulls rather than smoothing or
dropping groups.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as sk_metrics


METRIC_KEYS = (
    "balanced_accuracy",
    "resistant_recall",
    "susceptible_recall",
    "f1",
    "auroc",
    "pr_auc",
    "brier",
)


def _nullable(value: float) -> float | None:
    if np.isnan(value) or np.isinf(value):
        return None
    return float(value)


def score_binary(y_true: np.ndarray, y_prob: np.ndarray, calls: list[str]) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    calls_arr = np.asarray(calls, dtype=object)

    out: dict[str, Any] = {k: None for k in METRIC_KEYS}
    mask = np.isin(calls_arr, ["resistant", "susceptible"])
    if mask.any():
        y_pred = np.where(calls_arr[mask] == "resistant", 1, 0)
        yt = y_true[mask]
        tp = int(((yt == 1) & (y_pred == 1)).sum())
        tn = int(((yt == 0) & (y_pred == 0)).sum())
        fp = int(((yt == 0) & (y_pred == 1)).sum())
        fn = int(((yt == 1) & (y_pred == 0)).sum())
        recalls = []
        if (yt == 1).any():
            out["resistant_recall"] = _nullable(tp / (tp + fn))
            recalls.append(out["resistant_recall"])
        if (yt == 0).any():
            out["susceptible_recall"] = _nullable(tn / (tn + fp))
            recalls.append(out["susceptible_recall"])
        if recalls:
            out["balanced_accuracy"] = _nullable(float(np.mean(recalls)))
        denom = (2 * tp) + fp + fn
        out["f1"] = _nullable(0.0 if denom == 0 else (2 * tp) / denom)

    if len(y_true) > 0:
        out["brier"] = _nullable(sk_metrics.brier_score_loss(y_true, y_prob))
    if len(set(y_true.tolist())) == 2:
        out["auroc"] = _nullable(sk_metrics.roc_auc_score(y_true, y_prob))
        precision, recall, _ = sk_metrics.precision_recall_curve(y_true, y_prob)
        out["pr_auc"] = _nullable(sk_metrics.auc(recall, precision))
    return out


def no_call_stats(y_true: np.ndarray, final_calls: list[str]) -> dict[str, Any]:
    """Return the no-call rate and accuracy for predictions that were called."""
    yt = np.asarray(y_true, dtype=int)
    calls_arr = np.asarray(final_calls, dtype=object)
    n = int(calls_arr.size)
    out: dict[str, Any] = {"no_call_rate": None, "n_called": 0, "accuracy_on_called": None}
    if n == 0:
        return out
    out["no_call_rate"] = _nullable(float((calls_arr == "no_call").sum()) / n)
    called = np.isin(calls_arr, ["resistant", "susceptible"])
    out["n_called"] = int(called.sum())
    if called.any():
        y_pred = np.where(calls_arr[called] == "resistant", 1, 0)
        out["accuracy_on_called"] = _nullable(float((y_pred == yt[called]).mean()))
    return out
