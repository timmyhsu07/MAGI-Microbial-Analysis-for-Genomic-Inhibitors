"""End-to-end orchestration for Module 1.

Wires the stages together: discovery -> annotation -> parse -> feature matrix +
long table -> schema -> manifest, then writes all artifacts to the output
directory. The only external dependency (AMRFinderPlus) is injected via the
``runner`` argument so the entire pipeline is testable without the tool.

Row-set policy for the binary matrix: a genome is a matrix row iff it produced a
valid AMRFinderPlus TSV (status ``annotated`` or ``cached``) -- including
genomes with zero hits, which become all-zero rows. Genomes that were skipped
(protein) or failed are recorded in the manifest with their status and
deliberately excluded from the matrix, so an annotation failure is never
silently encoded as "no resistance genes".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import constants
from .annotate import Runner, annotate_genomes
from .config import Config, validate_config, with_run_id
from .discovery import discover_genomes
from .manifest import build_manifest, write_json
from .matrix import build_binary_matrix, build_long_table
from .parse import parse_tsv
from .schema import build_schema
from .versions import collect_versions


class PipelineError(RuntimeError):
    """Raised when the run cannot complete (e.g. genome failures, no matrix)."""


@dataclass(frozen=True)
class RunResult:
    run_id: str
    output_dir: str
    n_genomes_in_matrix: int
    n_features: int
    n_hits: int
    binary_matrix_path: str
    long_table_path: str
    schema_path: str
    manifest_path: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_run_id(genome_shas: list[str], cfg: Config, versions: dict[str, Any]) -> str:
    """Deterministic run id from inputs + the parameters that affect outputs."""
    amr = versions.get("amrfinderplus", {})
    h = hashlib.sha256()
    for s in sorted(genome_shas):
        h.update(s.encode())
    h.update(str(cfg.organism).encode())
    h.update(str(cfg.use_plus).encode())
    h.update(",".join(cfg.include_element_types).encode())
    h.update(str(amr.get("software_version")).encode())
    h.update(str(amr.get("database_version")).encode())
    return h.hexdigest()[:16]


def run_pipeline(
    cfg: Config,
    runner: Runner | None = None,
    require_amrfinder: bool = True,
    now_fn: Callable[[], str] = _utc_now_iso,
) -> RunResult:
    """Run the full Module 1 pipeline and write all artifacts.

    ``require_amrfinder`` fails loudly if the AMRFinderPlus version/database
    cannot be resolved. Tests that inject a mock ``runner`` pass False.
    ``now_fn`` is injectable so tests can freeze time for byte-identical output.
    """
    validate_config(cfg)  # fail before any side effects
    started_at = now_fn()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    versions = collect_versions(
        cfg.amrfinder_bin, cfg.database_dir, require_amrfinder=require_amrfinder
    )

    genomes = discover_genomes(cfg)
    run_id = cfg.run_id or _make_run_id([g.sha256 for g in genomes], cfg, versions)
    cfg = with_run_id(cfg, run_id)

    results = annotate_genomes(genomes, cfg, versions, runner=runner)

    failed = [r for r in results if r.status == "failed"]
    if failed and not cfg.allow_partial:
        detail = "\n  - ".join(f"{r.genome_id}: {r.error}" for r in failed)
        raise PipelineError(
            f"{len(failed)} genome(s) failed annotation and allow_partial is "
            f"False:\n  - {detail}"
        )

    # Parse TSVs; collect records and per-genome hit counts.
    all_records: list[dict[str, Any]] = []
    hits_by_genome: dict[str, int] = {}
    matrix_genome_ids: list[str] = []
    results_by_id = {r.genome_id: r for r in results}
    for g in genomes:  # sorted order
        r = results_by_id[g.genome_id]
        if not r.has_tsv:
            continue
        recs = parse_tsv(r.tsv_path, g.genome_id)
        all_records.extend(recs)
        hits_by_genome[g.genome_id] = len(recs)
        matrix_genome_ids.append(g.genome_id)

    if not matrix_genome_ids:
        raise PipelineError(
            "No genomes produced usable AMRFinderPlus output; nothing to build "
            "a matrix from. Check annotation status in the run manifest."
        )

    binary_df = build_binary_matrix(
        all_records, matrix_genome_ids, cfg.include_element_types
    )
    long_df = build_long_table(all_records)
    feature_columns = [c for c in binary_df.columns if c != constants.ID_COLUMN]

    schema = build_schema(
        all_records,
        feature_columns,
        matrix_genome_ids,
        cfg,
        versions,
        generated_at=now_fn(),
    )

    # --- write artifacts ---
    binary_parquet = out_dir / constants.OUT_BINARY_PARQUET
    binary_csv = out_dir / constants.OUT_BINARY_CSV
    long_parquet = out_dir / constants.OUT_LONG_PARQUET
    schema_path = out_dir / constants.OUT_SCHEMA_JSON
    manifest_path = out_dir / constants.OUT_MANIFEST_JSON

    binary_df.to_parquet(binary_parquet, index=False)
    binary_df.to_csv(binary_csv, index=False, lineterminator="\n")
    long_df.to_parquet(long_parquet, index=False)
    write_json(schema, schema_path)

    finished_at = now_fn()
    manifest = build_manifest(
        cfg=cfg,
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        versions=versions,
        genomes=genomes,
        results=results,
        hits_by_genome=hits_by_genome,
        matrix_genome_ids=matrix_genome_ids,
        feature_columns=feature_columns,
    )
    write_json(manifest, manifest_path)

    return RunResult(
        run_id=run_id,
        output_dir=str(out_dir),
        n_genomes_in_matrix=len(matrix_genome_ids),
        n_features=len(feature_columns),
        n_hits=sum(hits_by_genome.values()),
        binary_matrix_path=str(binary_parquet),
        long_table_path=str(long_parquet),
        schema_path=str(schema_path),
        manifest_path=str(manifest_path),
    )
