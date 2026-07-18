"""The load-bearing test for this module.

If a genetic cluster ever lands in both the training and the test half of any
fold, near-duplicate genomes leak across the split and every downstream metric
is inflated. This asserts that never happens — at the cluster level and at the
row level — and that the calibration hold-out is itself disjoint from both.
"""

from __future__ import annotations

import numpy as np

from predictor.splits import make_folds


def _clusters():
    # 24 genomes in 8 clusters of uneven size, deliberately unsorted/ragged so a
    # naive index split would leak.
    return np.array(
        [0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4, 4, 5, 6, 6, 6, 7, 7, 0, 2, 5, 4, 6, 3]
    )


def test_no_cluster_spans_train_and_test():
    clusters = _clusters()
    folds = make_folds(clusters, n_splits=4, calibration_fraction=0.25, seed=1)

    tested = set()
    for fold in folds:
        train_clusters = set(fold.fit_clusters) | set(fold.calib_clusters)
        test_clusters = set(fold.test_clusters)

        # The guarantee the whole submission rests on.
        assert train_clusters.isdisjoint(test_clusters), (
            f"cluster leak across train/test: {train_clusters & test_clusters}"
        )
        # Calibration must not overlap the model-fit set or the test set.
        assert set(fold.fit_clusters).isdisjoint(fold.calib_clusters)
        assert set(fold.calib_clusters).isdisjoint(test_clusters)
        tested |= test_clusters

    # Every cluster is tested exactly once across the folds.
    assert tested == set(clusters.tolist())


def test_row_indices_match_cluster_membership():
    clusters = _clusters()
    folds = make_folds(clusters, n_splits=4, calibration_fraction=0.25, seed=1)

    for fold in folds:
        # Row-level partitions are disjoint...
        assert set(fold.fit_idx).isdisjoint(fold.test_idx)
        assert set(fold.calib_idx).isdisjoint(fold.test_idx)
        assert set(fold.fit_idx).isdisjoint(fold.calib_idx)
        # ...and every row index carries the cluster its fold assigned it to.
        for idx in fold.test_idx:
            assert clusters[idx] in fold.test_clusters
        for idx in fold.fit_idx:
            assert clusters[idx] in fold.fit_clusters


def test_folds_are_deterministic():
    clusters = _clusters()
    a = make_folds(clusters, n_splits=4, calibration_fraction=0.25, seed=1)
    b = make_folds(clusters, n_splits=4, calibration_fraction=0.25, seed=1)
    assert [sorted(f.test_clusters) for f in a] == [sorted(f.test_clusters) for f in b]
