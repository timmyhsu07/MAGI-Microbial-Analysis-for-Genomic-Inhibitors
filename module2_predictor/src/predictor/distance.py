"""Genetic distance and single-linkage clustering.

The offline path uses Jaccard distance over the frozen binary AMR matrix. That
is a coarse proxy for whole-genome distance, but it keeps fixtures and CI
self-contained. A real Mash run is intentionally fail-fast when the binary or
sketch directory is absent so an operator never mistakes the proxy for Mash.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .config import DistanceConfig


class DistanceError(RuntimeError):
    """Raised when a configured distance backend cannot be used."""


def jaccard_distance_matrix(x: np.ndarray) -> np.ndarray:
    xb = np.asarray(x).astype(bool)
    n = xb.shape[0]
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        both = np.logical_and(xb[i], xb[i:]).sum(axis=1)
        either = np.logical_or(xb[i], xb[i:]).sum(axis=1)
        similarity = np.divide(
            both, either, out=np.zeros_like(both, dtype=float), where=either != 0
        )
        dist = 1.0 - similarity
        dist[either == 0] = 0.0
        out[i, i:] = dist
        out[i:, i] = dist
    return out


def nearest_jaccard(query: np.ndarray, train: np.ndarray) -> float | None:
    if len(train) == 0:
        return None
    q = np.asarray(query).astype(bool)
    t = np.asarray(train).astype(bool)
    both = np.logical_and(t, q).sum(axis=1)
    either = np.logical_or(t, q).sum(axis=1)
    similarity = np.divide(
        both, either, out=np.zeros_like(both, dtype=float), where=either != 0
    )
    dist = 1.0 - similarity
    dist[either == 0] = 0.0
    return float(np.min(dist))


def distance_matrix(x: np.ndarray, cfg: DistanceConfig) -> np.ndarray:
    if cfg.method == "jaccard":
        return jaccard_distance_matrix(x)
    if cfg.method == "mash":
        _require_mash(cfg)
        raise DistanceError(
            "distance.method='mash' is configured, but this training entrypoint "
            "needs a pairwise Mash distance table adapter for the provided "
            "sketch directory. Use distance.method='jaccard' for offline runs."
        )
    raise DistanceError(f"unsupported distance method: {cfg.method}")


def _require_mash(cfg: DistanceConfig) -> None:
    if shutil.which("mash") is None:
        raise DistanceError(
            "distance.method='mash' requires the 'mash' binary on PATH. "
            "Install Mash or set distance.method='jaccard' for offline fixtures."
        )
    if not cfg.mash_sketch_dir:
        raise DistanceError(
            "distance.method='mash' requires distance.mash_sketch_dir to point "
            "at precomputed genome sketches."
        )
    sketch_dir = Path(cfg.mash_sketch_dir)
    if not sketch_dir.exists() or not sketch_dir.is_dir():
        raise DistanceError(
            f"distance.mash_sketch_dir does not exist or is not a directory: {sketch_dir}"
        )


def cluster_from_distance(dist: np.ndarray, threshold: float) -> np.ndarray:
    if dist.shape[0] == 0:
        return np.array([], dtype=int)
    if dist.shape[0] == 1:
        return np.array([0], dtype=int)

    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="single")
    raw = fcluster(z, t=threshold, criterion="distance")

    # Dense ids by first appearance make manifests stable across scipy versions.
    remap: dict[int, int] = {}
    dense = []
    for c in raw.tolist():
        remap.setdefault(int(c), len(remap))
        dense.append(remap[int(c)])
    return np.array(dense, dtype=int)
