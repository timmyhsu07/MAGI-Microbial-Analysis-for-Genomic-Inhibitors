"""Training orchestration for Module 2.

The run deliberately separates statistical prediction from deterministic
overrides. Cross-validation is grouped by genetic cluster, calibration is fit
on a held-out cluster set rather than the test fold, and the decision log keeps
the model call, target gate, OOD guard, and final call distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .config import Config
from .distance import DistanceError, cluster_from_distance, distance_matrix, nearest_jaccard
from .gate import apply_gate
from .io import (
    ID_COLUMN,
    feature_columns_from_schema,
    load_inputs,
    matrix_values,
    write_json,
)
from .metrics import score_binary
from .model import ConstantProbabilityModel
from .report import write_reliability_plot
from .splits import Fold, make_folds


class TrainingError(RuntimeError):
    """Raised when the configured training run cannot complete."""


@dataclass(frozen=True)
class RunResult:
    output_dir: str
    n_genomes: int
    n_clusters: int
    trained_drugs: tuple[str, ...]
    skipped_drugs: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _library_versions() -> dict[str, str]:
    return {
        "numpy": _pkg_version("numpy"),
        "pandas": _pkg_version("pandas"),
        "pyarrow": _pkg_version("pyarrow"),
        "scikit-learn": _pkg_version("scikit-learn"),
        "scipy": _pkg_version("scipy"),
        "matplotlib": _pkg_version("matplotlib"),
        "joblib": _pkg_version("joblib"),
    }


def _fit_model(x: np.ndarray, y: np.ndarray, cfg: Config) -> Any:
    if len(set(y.tolist())) < 2:
        return ConstantProbabilityModel(float(y[0]) if len(y) else 0.5)
    model = LogisticRegression(
        C=cfg.model.C,
        max_iter=cfg.model.max_iter,
        class_weight="balanced",
        random_state=cfg.seed,
        solver="liblinear",
    )
    return model.fit(x, y)


def _resistant_probability(model: Any, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = getattr(model, "classes_", np.array([0, 1]))
    pos = int(np.where(classes == 1)[0][0])
    return proba[:, pos].astype(float)


def _fit_calibrator(raw_prob: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    if len(raw_prob) == 0:
        calibrator.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    else:
        calibrator.fit(raw_prob.astype(float), y.astype(float))
    return calibrator


def _call_from_probability(prob: float) -> str:
    return "resistant" if prob >= 0.5 else "susceptible"


def _final_decision(
    prob: float, model_call: str, gate_call: str | None, ood: bool, min_confidence: float
) -> tuple[str, str]:
    if gate_call is not None:
        return gate_call, "target_gate"
    if ood:
        return "no_call", "ood"
    if (1.0 - min_confidence) <= prob <= min_confidence:
        return "no_call", "low_confidence"
    return model_call, "model"


def _target_rows(targets: pd.DataFrame) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for rec in targets.to_dict(orient="records"):
        gid = str(rec.pop(ID_COLUMN))
        rows[gid] = {str(k): int(v) for k, v in rec.items()}
    return rows


def _labels_by_genome(labels: pd.DataFrame, genome_ids: list[str], drug: str) -> np.ndarray:
    by_id = labels.set_index(ID_COLUMN)[drug].to_dict()
    return np.array([str(by_id.get(gid, "")) for gid in genome_ids], dtype=object)


def _known_label_mask(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, ["R", "S"])


def _label_to_int(labels: np.ndarray) -> np.ndarray:
    return np.where(labels == "R", 1, 0).astype(int)


def _subset(indices: tuple[int, ...], mask: np.ndarray) -> np.ndarray:
    return np.array([i for i in indices if mask[i]], dtype=int)


def _nearest_training_distance(x: np.ndarray, train: np.ndarray, method: str) -> float | None:
    if method == "jaccard":
        return nearest_jaccard(x, train)
    raise DistanceError("OOD nearest-neighbour scoring currently supports jaccard runs")


def _fold_records(
    cfg: Config,
    drug: str,
    fold: Fold,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    known: np.ndarray,
    ood_reference_idx: np.ndarray,
    genome_ids: list[str],
    cluster_ids: np.ndarray,
    targets: dict[str, dict[str, int]],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[str]]:
    fit_idx = _subset(fold.fit_idx, known)
    calib_idx = _subset(fold.calib_idx, known)
    test_idx = _subset(fold.test_idx, known)
    if len(test_idx) == 0:
        return [], np.array([], dtype=int), np.array([], dtype=float), []
    if len(fit_idx) == 0:
        raise TrainingError(f"{drug}: fold has no labelled fit genomes")

    model = _fit_model(x[fit_idx], y[fit_idx], cfg)
    raw_calib = _resistant_probability(model, x[calib_idx]) if len(calib_idx) else np.array([])
    calibrator = _fit_calibrator(raw_calib, y[calib_idx])
    probs = calibrator.transform(_resistant_probability(model, x[test_idx]))

    records: list[dict[str, Any]] = []
    calls: list[str] = []
    target_cfg = cfg.drugs[drug]

    for idx, prob in zip(test_idx.tolist(), probs.tolist()):
        gid = genome_ids[idx]
        model_call = _call_from_probability(float(prob))
        gate_call = apply_gate(
            targets.get(gid, {}),
            target_cfg["target_genes"],
            target_cfg["on_target_absent"],
        )
        # OOF probability tests generalization; OOD mirrors the deployed guard's
        # representativeness check over all known-label training genomes.
        reference_idx = ood_reference_idx[ood_reference_idx != idx]
        reference_x = x[reference_idx] if len(reference_idx) else np.empty((0, x.shape[1]))
        nearest = _nearest_training_distance(x[idx], reference_x, cfg.distance.method)
        is_ood = nearest is not None and nearest > cfg.ood.threshold
        final_call, source = _final_decision(
            float(prob), model_call, gate_call, is_ood, cfg.decision.min_confidence
        )
        calls.append(final_call)
        records.append(
            {
                "genome_id": gid,
                "drug": drug,
                "label": labels[idx],
                "cluster_id": int(cluster_ids[idx]),
                "calibrated_prob": float(prob),
                "model_call": model_call,
                "gate_call": gate_call or "",
                "ood": bool(is_ood),
                "nearest_train_distance": nearest,
                "final_call": final_call,
                "decision_source": source,
            }
        )
    return records, y[test_idx], np.asarray(probs, dtype=float), calls


def _metric_entry(
    records: list[dict[str, Any]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    calls: list[str],
    n_clusters: int,
) -> dict[str, Any]:
    per_group: dict[str, Any] = {}
    rec_df = pd.DataFrame(records)
    if not rec_df.empty:
        for cluster_id, group in rec_df.groupby("cluster_id", sort=True):
            idx = group.index.to_numpy()
            per_group[str(int(cluster_id))] = {
                "n": int(len(group)),
                **score_binary(y_true[idx], y_prob[idx], group["model_call"].tolist()),
            }
    return {
        "n_genomes": int(len(records)),
        "n_clusters": int(n_clusters),
        "overall": score_binary(
            y_true, y_prob, pd.DataFrame(records)["model_call"].tolist()
        ),
        "per_group": per_group,
    }


def _choose_final_split(
    known_idx: np.ndarray, cluster_ids: np.ndarray, y: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    clusters = np.array(sorted({int(cluster_ids[i]) for i in known_idx.tolist()}), dtype=int)
    rng.shuffle(clusters)

    if len(clusters) <= 1:
        return known_idx, np.array([], dtype=int)

    n_calib = min(max(1, int(round(len(clusters) * fraction))), len(clusters) - 1)
    calib_clusters = set(int(c) for c in clusters[:n_calib].tolist())

    def split() -> tuple[np.ndarray, np.ndarray]:
        calib = np.array([i for i in known_idx.tolist() if int(cluster_ids[i]) in calib_clusters], dtype=int)
        fit = np.array([i for i in known_idx.tolist() if int(cluster_ids[i]) not in calib_clusters], dtype=int)
        return fit, calib

    fit_idx, calib_idx = split()
    for c in list(calib_clusters):
        if len(set(y[fit_idx].tolist())) == 2:
            break
        calib_clusters.remove(c)
        fit_idx, calib_idx = split()

    if len(calib_idx) == 0:
        calib_idx = fit_idx
    return fit_idx, calib_idx


def _write_artifacts(
    cfg: Config,
    drug: str,
    out_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    known_idx: np.ndarray,
    cluster_ids: np.ndarray,
    feature_columns: list[str],
    feature_schema: dict[str, Any],
    versions: dict[str, str],
) -> None:
    fit_idx, calib_idx = _choose_final_split(
        known_idx, cluster_ids, y, cfg.cv.calibration_fraction, cfg.seed
    )
    model = _fit_model(x[fit_idx], y[fit_idx], cfg)
    calibrator = _fit_calibrator(_resistant_probability(model, x[calib_idx]), y[calib_idx])

    coef = getattr(model, "coef_", np.zeros((1, len(feature_columns)), dtype=float))
    if coef.shape[1] != len(feature_columns):
        coef = np.zeros((1, len(feature_columns)), dtype=float)
    intercept = getattr(model, "intercept_", np.array([0.0], dtype=float))
    train_fingerprints = x[known_idx].astype(int).tolist()
    target_cfg = cfg.drugs[drug]

    artifact = {
        "schema_version": "1.0.0",
        "drug": drug,
        "trained_at": _utc_now_iso(),
        "positive_class": "R",
        "feature_columns": feature_columns,
        "model": {
            "kind": "logistic_regression",
            "penalty": "l2",
            "C": float(cfg.model.C),
            "class_weight": "balanced",
            "coef": [float(v) for v in coef[0].tolist()],
            "intercept": float(intercept[0]),
        },
        "calibrator": {
            "kind": "isotonic",
            "x_thresholds": [float(v) for v in calibrator.X_thresholds_.tolist()],
            "y_thresholds": [float(v) for v in calibrator.y_thresholds_.tolist()],
        },
        "gate": {
            "target_genes": target_cfg["target_genes"],
            "on_target_absent": target_cfg["on_target_absent"],
        },
        "decision": {"min_confidence": float(cfg.decision.min_confidence)},
        "ood": {
            "threshold": float(cfg.ood.threshold),
            "train_fingerprints": train_fingerprints,
        },
        "provenance": {
            "seed": int(cfg.seed),
            "n_train_genomes": int(len(known_idx)),
            "n_train_clusters": int(len({int(cluster_ids[i]) for i in known_idx.tolist()})),
            "cluster_threshold": float(cfg.distance.cluster_threshold),
            "distance_method": cfg.distance.method,
            "feature_schema_version": str(feature_schema.get("schema_version", "unknown")),
            "library_versions": versions,
        },
    }
    write_json(artifact, out_dir / "models" / f"{drug}.model.json")
    joblib.dump(
        {
            "drug": drug,
            "model": model,
            "calibrator": calibrator,
            "feature_columns": feature_columns,
            "gate": artifact["gate"],
            "decision": artifact["decision"],
            "ood": artifact["ood"],
            "feature_schema_version": artifact["provenance"]["feature_schema_version"],
        },
        out_dir / "models" / f"{drug}.joblib",
    )


def run(cfg: Config) -> RunResult:
    out_dir = Path(cfg.output_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "reliability").mkdir(parents=True, exist_ok=True)

    features, feature_schema, labels_df, targets_df = load_inputs(cfg)
    feature_columns = feature_columns_from_schema(feature_schema)
    genome_ids = [str(g) for g in features[ID_COLUMN].tolist()]
    x = matrix_values(features, feature_columns)

    dist = distance_matrix(x, cfg.distance)
    cluster_ids = cluster_from_distance(dist, cfg.distance.cluster_threshold)
    folds = make_folds(cluster_ids, cfg.cv.n_splits, cfg.cv.calibration_fraction, cfg.seed)
    targets = _target_rows(targets_df)
    versions = _library_versions()

    all_records: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "drugs": {},
        "skipped_drugs": {},
        "config": cfg.to_serializable(),
        "versions": versions,
    }
    trained: list[str] = []

    for drug in sorted(cfg.drugs):
        label_values = _labels_by_genome(labels_df, genome_ids, drug)
        known = _known_label_mask(label_values)
        known_idx = np.where(known)[0]
        y = _label_to_int(label_values)
        counts = {
            "R": int(((label_values == "R") & known).sum()),
            "S": int(((label_values == "S") & known).sum()),
        }
        if counts["R"] < cfg.cv.min_per_class or counts["S"] < cfg.cv.min_per_class:
            report["skipped_drugs"][drug] = {
                "reason": "too_few_labelled_genomes_per_class",
                "class_counts": counts,
            }
            continue

        records: list[dict[str, Any]] = []
        y_true_parts: list[np.ndarray] = []
        y_prob_parts: list[np.ndarray] = []
        calls: list[str] = []
        for fold in folds:
            fold_records, yt, yp, fold_calls = _fold_records(
                cfg,
                drug,
                fold,
                x,
                y,
                label_values,
                known,
                known_idx,
                genome_ids,
                cluster_ids,
                targets,
            )
            records.extend(fold_records)
            if len(yt):
                y_true_parts.append(yt)
                y_prob_parts.append(yp)
                calls.extend(fold_calls)

        if not records:
            report["skipped_drugs"][drug] = {
                "reason": "no_out_of_fold_predictions",
                "class_counts": counts,
            }
            continue

        y_true = np.concatenate(y_true_parts)
        y_prob = np.concatenate(y_prob_parts)
        n_clusters = len({int(cluster_ids[i]) for i in np.where(known)[0].tolist()})
        report["drugs"][drug] = _metric_entry(records, y_true, y_prob, calls, n_clusters)
        all_records.extend(records)
        write_reliability_plot(y_true, y_prob, out_dir / "reliability" / f"{drug}.png")
        _write_artifacts(
            cfg,
            drug,
            out_dir,
            x,
            y,
            known_idx,
            cluster_ids,
            feature_columns,
            feature_schema,
            versions,
        )
        trained.append(drug)

    pd.DataFrame(all_records).sort_values(["drug", "genome_id"]).to_csv(
        out_dir / "decisions.csv", index=False, lineterminator="\n"
    )
    write_json(report, out_dir / "metrics_report.json")
    write_json(
        {
            "seed": int(cfg.seed),
            "config": cfg.to_serializable(),
            "library_versions": versions,
            "clusters": [
                {"genome_id": gid, "cluster_id": int(cid)}
                for gid, cid in zip(genome_ids, cluster_ids.tolist())
            ],
            "folds": [
                {
                    "fold": i,
                    "fit_clusters": list(f.fit_clusters),
                    "calib_clusters": list(f.calib_clusters),
                    "test_clusters": list(f.test_clusters),
                }
                for i, f in enumerate(folds)
            ],
        },
        out_dir / "run_manifest.json",
    )

    return RunResult(
        output_dir=str(out_dir),
        n_genomes=len(genome_ids),
        n_clusters=len(set(cluster_ids.tolist())),
        trained_drugs=tuple(trained),
        skipped_drugs=report["skipped_drugs"],
    )
