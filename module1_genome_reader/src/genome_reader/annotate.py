"""Stage 2 - Annotation.

Run AMRFinderPlus once per genome, in parallel across a configurable number of
workers, writing one TSV per genome into a cache directory. Reruns skip genomes
whose cached result is still valid.

The actual invocation of AMRFinderPlus is isolated behind the ``Runner``
protocol. The production :class:`AmrfinderRunner` shells out to the tool; tests
inject a mock runner that writes canned TSVs, so the whole pipeline is
exercisable end-to-end without AMRFinderPlus installed.

Cache validity is keyed on everything that can change a result: the input file
checksum, the organism, the ``--plus`` setting, the sequence mode, and the
AMRFinderPlus software + database versions. Any change invalidates the cache
and forces a re-run, which keeps "same inputs -> identical outputs" honest.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import Config
from .discovery import Genome

CACHE_SCHEMA = 1


class AnnotationError(RuntimeError):
    """Raised when AMRFinderPlus fails for a genome."""


@dataclass(frozen=True)
class AnnotationResult:
    genome_id: str
    status: str  # annotated | cached | skipped_protein | failed
    tsv_path: str | None
    error: str | None = None
    duration_seconds: float | None = None

    @property
    def has_tsv(self) -> bool:
        return self.tsv_path is not None


class Runner(Protocol):
    """Produces an AMRFinderPlus TSV for a genome at ``out_tsv``.

    Implementations must either write a valid TSV to ``out_tsv`` or raise.
    ``mode`` is "nucleotide" or "protein".
    """

    def run(self, genome: Genome, out_tsv: Path, mode: str) -> None: ...


class AmrfinderRunner:
    """Runs the real AMRFinderPlus binary."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _decompress_if_needed(self, genome: Genome, work_dir: Path) -> Path:
        if genome.path.suffix != ".gz":
            return genome.path
        dest = work_dir / f"{genome.genome_id}.input.fasta"
        with gzip.open(genome.path, "rb") as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        return dest

    def build_command(self, input_path: Path, out_tsv: Path, mode: str) -> list[str]:
        cfg = self.cfg
        flag = "-n" if mode == "nucleotide" else "-p"
        cmd = [cfg.amrfinder_bin, flag, str(input_path), "-o", str(out_tsv)]
        if cfg.organism:
            cmd += ["--organism", cfg.organism]
        if cfg.use_plus:
            cmd += ["--plus"]
        if cfg.database_dir:
            cmd += ["-d", cfg.database_dir]
        if cfg.amrfinder_threads:
            cmd += ["--threads", str(cfg.amrfinder_threads)]
        return cmd

    def run(self, genome: Genome, out_tsv: Path, mode: str) -> None:
        work_dir = out_tsv.parent
        input_path = self._decompress_if_needed(genome, work_dir)
        cmd = self.build_command(input_path, out_tsv, mode)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError as exc:
            raise AnnotationError(
                f"Failed to launch AMRFinderPlus for '{genome.genome_id}': {exc}. "
                f"Command: {' '.join(cmd)}"
            ) from exc
        if proc.returncode != 0:
            raise AnnotationError(
                f"AMRFinderPlus exited {proc.returncode} for '{genome.genome_id}'.\n"
                f"Command: {' '.join(cmd)}\nstderr:\n{proc.stderr}"
            )
        if not out_tsv.exists():
            raise AnnotationError(
                f"AMRFinderPlus reported success but wrote no output for "
                f"'{genome.genome_id}' (expected {out_tsv})"
            )


def _cache_meta_path(tsv_path: Path) -> Path:
    return tsv_path.with_suffix(".meta.json")


def _expected_meta(genome: Genome, cfg: Config, versions: dict[str, Any], mode: str) -> dict[str, Any]:
    amr = versions.get("amrfinderplus", {})
    return {
        "cache_schema": CACHE_SCHEMA,
        "genome_id": genome.genome_id,
        "input_sha256": genome.sha256,
        "mode": mode,
        "organism": cfg.organism,
        "use_plus": cfg.use_plus,
        "amrfinder_software_version": amr.get("software_version"),
        "amrfinder_database_version": amr.get("database_version"),
    }


def _cache_is_valid(tsv_path: Path, expected: dict[str, Any]) -> bool:
    meta_path = _cache_meta_path(tsv_path)
    if not (tsv_path.exists() and meta_path.exists()):
        return False
    try:
        actual = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def _annotate_one(
    genome: Genome,
    cfg: Config,
    versions: dict[str, Any],
    runner: Runner,
    cache_dir: Path,
) -> AnnotationResult:
    mode = "nucleotide" if genome.is_nucleotide else "protein"

    if not genome.is_nucleotide and cfg.protein_handling == "skip":
        return AnnotationResult(
            genome_id=genome.genome_id,
            status="skipped_protein",
            tsv_path=None,
            error=None,
        )

    tsv_path = cache_dir / f"{genome.genome_id}.amrfinder.tsv"
    expected = _expected_meta(genome, cfg, versions, mode)

    if cfg.reuse_cache and _cache_is_valid(tsv_path, expected):
        return AnnotationResult(
            genome_id=genome.genome_id,
            status="cached",
            tsv_path=str(tsv_path),
        )

    # Write to a temp path then atomically move, so an interrupted run never
    # leaves a half-written TSV that a later run would trust.
    tmp_tsv = tsv_path.with_suffix(".tsv.partial")
    if tmp_tsv.exists():
        tmp_tsv.unlink()
    start = time.monotonic()
    try:
        runner.run(genome, tmp_tsv, mode)
    except Exception as exc:  # noqa: BLE001 - report every failure as a result
        if tmp_tsv.exists():
            tmp_tsv.unlink()
        return AnnotationResult(
            genome_id=genome.genome_id,
            status="failed",
            tsv_path=None,
            error=str(exc),
            duration_seconds=round(time.monotonic() - start, 3),
        )

    tmp_tsv.replace(tsv_path)
    _cache_meta_path(tsv_path).write_text(
        json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8"
    )
    return AnnotationResult(
        genome_id=genome.genome_id,
        status="annotated",
        tsv_path=str(tsv_path),
        duration_seconds=round(time.monotonic() - start, 3),
    )


def annotate_genomes(
    genomes: list[Genome],
    cfg: Config,
    versions: dict[str, Any],
    runner: Runner | None = None,
) -> list[AnnotationResult]:
    """Annotate all genomes, returning one result per genome sorted by ID.

    Genome-level failures are captured in the returned results (status
    "failed"); the caller decides whether a partial corpus is acceptable. This
    lets one malformed genome fail without discarding the rest of the run.
    """
    runner = runner or AmrfinderRunner(cfg)
    cache_dir = cfg.resolved_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, AnnotationResult] = {}
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {
            pool.submit(_annotate_one, g, cfg, versions, runner, cache_dir): g.genome_id
            for g in genomes
        }
        for fut in as_completed(futures):
            result = fut.result()
            results[result.genome_id] = result

    return [results[g.genome_id] for g in genomes]
