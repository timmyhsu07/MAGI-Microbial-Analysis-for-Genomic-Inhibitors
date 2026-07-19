"""Real integration adapter: Module 1's real output + Module 2's real trained
artifacts, wired into Module 3's FeatureExtractor / Predictor protocols.

``mock_pipeline.py`` exists so Module 3 runs standalone with zero real
artifacts. This module is the other half of that story: once Module 1 has
produced a real ``output_dir`` (``features_binary`` + ``feature_schema`` [+
``features_long``]) and Module 2 has trained real per-drug model artifacts,
THIS is what a caller wires into :func:`decision_report.report.build_report`
instead of the mocks -- see ``README.md``'s "Wiring it to the real pipeline"
section.

Nothing here reimplements Module 2's math. Calibrated probability and the
target-gate call are read via :func:`predictor.inference.predict_one_genome` --
the exact function Module 2 itself uses at inference time -- not recomputed.
Only the per-drug feature-importance breakdown and a continuous
distance-to-nearest-training-genome are computed here (from the stored model
coefficients and :func:`predictor.distance.nearest_jaccard`), because Module 3
owns the OOD *threshold decision*; Module 2 already reduced that to a boolean
for its own reporting, which is not enough for Module 3's uncertainty rules.

The ``predictor`` package (Module 2) is imported lazily inside functions, so
``decision_report`` has no hard install-time dependency on it -- consistent
with how ``config.py`` already lazy-imports ``yaml``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import DecisionReportError, DrugPrediction, FeatureBundle, LongHit, TopFeature

ID_COLUMN = "genome_id"

_KNOWN_EXTENSIONS = (".fasta.gz", ".fa.gz", ".fna.gz", ".fasta", ".fa", ".fna", ".fas", ".seq")


class IntegrationError(DecisionReportError):
    """Raised when the real Module 1/2 artifacts can't be wired up."""


def _genome_id_from_identifier(identifier: str) -> str:
    """Accept either a bare genome_id or a FASTA path; return the genome_id."""
    name = Path(identifier).name
    lower = name.lower()
    for ext in _KNOWN_EXTENSIONS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return name


class Module1FeatureStore:
    """Loads a real Module 1 output directory once; serves FeatureBundle lookups.

    Prefers the real per-hit provenance in ``features_long.parquet`` when
    present. If a directory only carries the binary matrix -- e.g. Module 2's
    fixture corpus, which is shaped exactly like Module 1 output but never ran
    AMRFinderPlus -- hits are reconstructed from ``feature_schema.json``'s
    per-column metadata instead: one synthesized hit per present feature,
    lower-fidelity than a true per-hit record, but enough to drive evidence
    categorization. ``used_reconstructed_hits`` tells a caller which path was
    taken so this fallback is never silently mistaken for real provenance.
    """

    def __init__(self, output_dir: str | Path):
        out = Path(output_dir)
        self._binary_by_id = self._read_binary_matrix(out)
        long_path = out / "features_long.parquet"
        self._long = pd.read_parquet(long_path) if long_path.exists() else None
        schema_path = out / "feature_schema.json"
        if not schema_path.exists():
            raise IntegrationError(f"{out}: missing feature_schema.json")
        self._schema_by_column = {
            c["column"]: c for c in json.loads(schema_path.read_text(encoding="utf-8"))["columns"]
        }
        self.used_reconstructed_hits = self._long is None

    def genome_ids(self) -> list[str]:
        """Genome IDs present in this Module 1 output directory, sorted -- lets
        the UI offer a dropdown instead of a free-text ID box (no typing on
        stage, no guessing identifiers that aren't in the store)."""
        return sorted(self._binary_by_id)

    @staticmethod
    def _read_binary_matrix(out: Path) -> dict[str, dict[str, int]]:
        parquet_path = out / "features_binary.parquet"
        csv_path = out / "features_binary.csv"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, keep_default_na=False)
        else:
            raise IntegrationError(f"{out}: missing features_binary.parquet/.csv")
        return {
            str(row[ID_COLUMN]): {k: int(v) for k, v in row.items() if k != ID_COLUMN}
            for row in df.to_dict(orient="records")
        }

    def _hits_from_long_table(self, genome_id: str) -> list[LongHit]:
        rows = self._long[self._long[ID_COLUMN] == genome_id]
        return [
            LongHit(
                element_symbol=rec.get("element_symbol"),
                element_type=rec.get("element_type") or "AMR",
                element_subtype=rec.get("element_subtype") or "",
                method=rec.get("method") or "",
                drug_class=rec.get("class"),
                pct_identity=rec.get("pct_identity"),
                pct_coverage=rec.get("pct_coverage"),
            )
            for rec in rows.to_dict(orient="records")
        ]

    def _hits_from_schema(self, binary_row: dict[str, int]) -> list[LongHit]:
        hits = []
        for column, present in binary_row.items():
            if not present:
                continue
            meta = self._schema_by_column.get(column)
            if meta is None:
                continue
            drug_classes = meta.get("drug_classes") or []
            methods = meta.get("methods_observed") or []
            subtypes = meta.get("element_subtypes_observed") or []
            hits.append(
                LongHit(
                    element_symbol=column,
                    element_type=meta.get("element_type") or "AMR",
                    element_subtype=subtypes[0] if subtypes else "",
                    method=methods[0] if methods else "",
                    drug_class=drug_classes[0] if drug_classes else None,
                )
            )
        return hits

    def __call__(self, identifier: str) -> FeatureBundle:
        genome_id = _genome_id_from_identifier(identifier)
        binary_row = self._binary_by_id.get(genome_id)
        if binary_row is None:
            raise IntegrationError(
                f"genome '{genome_id}' not found in this Module 1 output directory"
            )
        hits = (
            self._hits_from_long_table(genome_id)
            if self._long is not None
            else self._hits_from_schema(binary_row)
        )
        return FeatureBundle(genome_id=genome_id, binary_row=binary_row, long_hits=hits)


class ModelPredictor:
    """Predictor protocol backed by Module 2's real trained artifacts.

    ``covered_drugs()`` is DERIVED from whatever models are actually on disk,
    never hardcoded -- it can never drift out of sync with what was trained,
    which is what let the mock demo and the real config disagree on the
    covered drug set (see AUDIT.md).
    """

    def __init__(self, models_dir: str | Path, target_gene_table: str | Path, species: str):
        try:
            from predictor.inference import load_artifacts
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise IntegrationError(
                "the 'predictor' package (Module 2) is not installed in this "
                "environment; install it with 'pip install -e ../module2_predictor' "
                "to use the real pipeline"
            ) from exc

        self._artifacts = load_artifacts(models_dir)
        if not self._artifacts:
            raise IntegrationError(f"no trained model artifacts found in {models_dir}")
        targets_df = pd.read_csv(target_gene_table, keep_default_na=False)
        self._targets_by_id = {
            str(row[ID_COLUMN]): {k: int(v) for k, v in row.items() if k != ID_COLUMN}
            for row in targets_df.to_dict(orient="records")
        }
        self._species = species

    def covered_drugs(self) -> list[str]:
        return sorted(self._artifacts)

    def covered_species(self) -> list[str]:
        return [self._species]

    def ood_threshold(self) -> float:
        """The OOD threshold Module 2 was trained with, for a caller to pass
        straight into Module 3's DecisionConfig instead of hand-copying a
        second value that can drift out of sync with the trained models."""
        thresholds = {float(a["ood"]["threshold"]) for a in self._artifacts.values()}
        return next(iter(thresholds)) if len(thresholds) == 1 else min(thresholds)

    def _top_features(
        self, artifact: dict[str, Any], binary_row: dict[str, int], n: int = 5
    ) -> list[TopFeature]:
        columns = artifact["feature_columns"]
        coef = getattr(artifact["model"], "coef_", None)
        if coef is None or coef.shape[1] != len(columns):
            return []  # e.g. ConstantProbabilityModel, fit when only one class was labelled
        contributions = [
            (col, float(coef[0][i]) * int(binary_row.get(col, 0))) for i, col in enumerate(columns)
        ]
        contributions = [c for c in contributions if c[1] != 0.0]
        contributions.sort(key=lambda c: abs(c[1]), reverse=True)
        return [TopFeature(name=name, contribution=contrib) for name, contrib in contributions[:n]]

    def _ood_distance(self, artifact: dict[str, Any], binary_row: dict[str, int]) -> float:
        from predictor.distance import nearest_jaccard

        columns = artifact["feature_columns"]
        x = np.array([int(binary_row.get(c, 0)) for c in columns], dtype=np.int8)
        train = np.asarray(artifact["ood"]["train_fingerprints"], dtype=np.int8)
        nearest = nearest_jaccard(x, train)
        return float(nearest) if nearest is not None else 0.0

    def predict(self, features: FeatureBundle) -> list[DrugPrediction]:
        from predictor.inference import predict_one_genome

        target_row = self._targets_by_id.get(features.genome_id, {})
        raw = predict_one_genome(features.binary_row, target_row, self._artifacts)
        predictions = []
        for drug in self.covered_drugs():
            r = raw[drug]
            artifact = self._artifacts[drug]
            predictions.append(
                DrugPrediction(
                    drug=drug,
                    calibrated_prob_resistant=r["calibrated_prob"],
                    target_present=r["gate_call"] is None,
                    top_features=self._top_features(artifact, features.binary_row),
                    ood_score=self._ood_distance(artifact, features.binary_row),
                )
            )
        return predictions
