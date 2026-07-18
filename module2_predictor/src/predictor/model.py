"""Small estimator helpers used when data are thinner than the happy path.

The configured training path uses logistic regression and isotonic calibration.
These wrappers keep edge cases importable and serializable without changing the
public artifact shape expected by downstream modules.
"""

from __future__ import annotations

import numpy as np


class ConstantProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)
        self.classes_ = np.array([0, 1], dtype=int)
        self.coef_ = np.zeros((1, 0), dtype=float)
        self.intercept_ = np.array([0.0], dtype=float)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])
