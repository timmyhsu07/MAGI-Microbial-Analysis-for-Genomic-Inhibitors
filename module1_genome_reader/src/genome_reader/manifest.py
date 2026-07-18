"""Run manifest assembly (run_manifest.json).

The manifest is the reproducibility record for a run: tool/database/dependency
versions, the fully-resolved config, per-genome input checksums and processing
status, and summary counts. It is provenance, not a model input -- so it is the
one artifact whose timestamps and per-genome durations legitimately vary
between otherwise-identical runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import constants
from .annotate import AnnotationResult
from .config import Config
from .discovery import Genome


def write_json(obj: dict[str, Any], path: Path) -> None:
    """Write JSON deterministically (stable key order as constructed)."""
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_manifest(
    cfg: Config,
    run_id: str,
    started_at: str,
    finished_at: str,
    versions: dict[str, Any],
    genomes: list[Genome],
    results: list[AnnotationResult],
    hits_by_genome: dict[str, int],
    matrix_genome_ids: list[str],
    feature_columns: list[str],
) -> dict[str, Any]:
    results_by_id = {r.genome_id: r for r in results}
    matrix_set = set(matrix_genome_ids)

    inputs = []
    status_counts: dict[str, int] = {}
    n_hits_total = 0
    for g in genomes:  # already sorted by id
        r = results_by_id[g.genome_id]
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
        n_hits = hits_by_genome.get(g.genome_id, 0)
        n_hits_total += n_hits
        inputs.append(
            {
                "genome_id": g.genome_id,
                "path": str(g.path),
                "sha256": g.sha256,
                "bytes": g.n_bytes,
                "seq_type": g.seq_type,
                "n_sequences": g.n_sequences,
                "status": r.status,
                "included_in_matrix": g.genome_id in matrix_set,
                "n_hits": n_hits,
                "annotation_seconds": r.duration_seconds,
                "error": r.error,
            }
        )

    return {
        "run_id": run_id,
        "pipeline_version": constants.PIPELINE_VERSION,
        "schema_version": constants.SCHEMA_VERSION,
        "started_at": started_at,
        "finished_at": finished_at,
        "config": cfg.to_serializable(),
        "tools": versions,
        "counts": {
            "n_genomes_discovered": len(genomes),
            "n_genomes_in_matrix": len(matrix_genome_ids),
            "n_features": len(feature_columns),
            "n_hits_total": n_hits_total,
            "status_breakdown": dict(sorted(status_counts.items())),
        },
        "inputs": inputs,
    }
