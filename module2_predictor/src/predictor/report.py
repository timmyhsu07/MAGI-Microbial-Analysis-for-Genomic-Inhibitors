"""Plotting utilities for model reliability artifacts."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile

import numpy as np


def write_reliability_plot(y_true: np.ndarray, y_prob: np.ndarray, path: str | Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, 11)
    mids = (bins[:-1] + bins[1:]) / 2
    observed = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        observed.append(float(y_true[mask].mean()) if mask.any() else np.nan)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.plot([0, 1], [0, 1], color="#888888", linewidth=1, linestyle="--")
    ax.plot(mids, observed, marker="o", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed resistance")
    ax.set_title("Reliability")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
