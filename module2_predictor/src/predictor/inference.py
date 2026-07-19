"""Inference helpers for downstream modules.

Prediction maps a query genome onto the frozen feature columns saved at
training time. Unknown features are ignored by construction; missing frozen
features are treated as absent, matching Module 1's presence/absence contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .distance import nearest_jaccard
from .gate import apply_gate


def load_artifacts(models_dir: str | Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(models_dir).glob("*.joblib")):
        artifact = joblib.load(path)
        artifacts[str(artifact["drug"])] = artifact
    return artifacts


def _vectorize(feature_row: Mapping[str, int], columns: list[str]) -> np.ndarray:
    return np.array([[int(feature_row.get(c, 0)) for c in columns]], dtype=np.int8)


def _model_probability(artifact: dict[str, Any], x: np.ndarray) -> float:
    model = artifact["model"]
    proba = model.predict_proba(x)
    classes = getattr(model, "classes_", np.array([0, 1]))
    pos = int(np.where(classes == 1)[0][0])
    return float(artifact["calibrator"].transform([float(proba[0, pos])])[0])


def _call(prob: float) -> str:
    return "resistant" if prob >= 0.5 else "susceptible"


def _evidence(feature_row: Mapping[str, int], columns: list[str], model_call: str) -> str:
    present = [c for c in columns if int(feature_row.get(c, 0)) == 1]
    if present:
        return "known_resistance_gene_or_point_mutation"
    if model_call == "resistant":
        return "statistical_only"
    return "no_signal"


def predict_one_genome(
    feature_row: Mapping[str, int],
    target_row: Mapping[str, int],
    artifacts: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for drug, artifact in sorted(artifacts.items()):
        columns = list(artifact["feature_columns"])
        x = _vectorize(feature_row, columns)
        prob = _model_probability(artifact, x)
        model_call = _call(prob)

        gate = artifact["gate"]
        gate_call = apply_gate(target_row, gate["target_genes"], gate["on_target_absent"])
        nearest = nearest_jaccard(x[0], np.asarray(artifact["ood"]["train_fingerprints"], dtype=np.int8))
        ood = nearest is not None and nearest > float(artifact["ood"]["threshold"])
        min_conf = float(artifact["decision"]["min_confidence"])

        if gate_call is not None:
            final_call, source = gate_call, "target_gate"
        elif ood:
            final_call, source = "no_call", "ood"
        elif (1.0 - min_conf) <= prob <= min_conf:
            final_call, source = "no_call", "low_confidence"
        else:
            final_call, source = model_call, "model"

        results[drug] = {
            "model_call": model_call,
            "calibrated_confidence": prob if model_call == "resistant" else 1.0 - prob,
            "calibrated_prob": prob,
            "gate_call": gate_call,
            "ood": bool(ood),
            "final_call": final_call,
            "decision_source": source,
            "evidence_category": _evidence(feature_row, columns, model_call),
        }
    return results
