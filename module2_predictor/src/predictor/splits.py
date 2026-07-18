"""Cluster-aware cross-validation splits.

Rows are never split directly because near-duplicate genomes can make ordinary
K-fold evaluation look better than it is. The caller supplies the cluster id for
each row; this module partitions clusters first, then expands back to row
indices for model code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    fit_clusters: tuple[int, ...]
    calib_clusters: tuple[int, ...]
    test_clusters: tuple[int, ...]
    fit_idx: tuple[int, ...]
    calib_idx: tuple[int, ...]
    test_idx: tuple[int, ...]


def _row_indices(cluster_ids: np.ndarray, clusters: tuple[int, ...]) -> tuple[int, ...]:
    selected = set(clusters)
    return tuple(int(i) for i, c in enumerate(cluster_ids) if c in selected)


def _calibration_count(n_clusters: int, fraction: float) -> int:
    if n_clusters == 0 or fraction <= 0:
        return 0
    return min(max(1, int(round(n_clusters * fraction))), n_clusters)


def make_folds(
    cluster_ids: np.ndarray, n_splits: int, calibration_fraction: float, seed: int
) -> list[Fold]:
    cluster_ids = np.asarray(cluster_ids)
    unique = np.array(sorted({int(c) for c in cluster_ids.tolist()}), dtype=int)
    if len(unique) == 0:
        return []

    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_folds = min(max(1, int(n_splits)), len(shuffled))

    folds: list[Fold] = []
    for test in np.array_split(shuffled, n_folds):
        test_clusters = tuple(sorted(int(c) for c in test.tolist()))
        remaining = np.array([c for c in shuffled if c not in set(test_clusters)], dtype=int)

        n_calib = _calibration_count(len(remaining), calibration_fraction)
        calib_clusters = tuple(sorted(int(c) for c in remaining[:n_calib].tolist()))
        fit_clusters = tuple(sorted(int(c) for c in remaining[n_calib:].tolist()))

        folds.append(
            Fold(
                fit_clusters=fit_clusters,
                calib_clusters=calib_clusters,
                test_clusters=test_clusters,
                fit_idx=_row_indices(cluster_ids, fit_clusters),
                calib_idx=_row_indices(cluster_ids, calib_clusters),
                test_idx=_row_indices(cluster_ids, test_clusters),
            )
        )
    return folds
