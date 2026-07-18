"""Stage 1 - Discovery.

Scan the configured input directory, derive a stable genome ID from each
filename, and validate every FASTA (parseable, non-empty, nucleotide vs
protein). Discovery is deterministic: genomes are returned sorted by ID, and a
duplicate ID (two files mapping to the same ID) is a loud error rather than a
silent last-writer-wins.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .fasta import FastaError, FastaStats, validate_fasta


class DiscoveryError(ValueError):
    """Raised when the input directory or a filename cannot be handled."""


@dataclass(frozen=True)
class Genome:
    """A discovered, validated input genome."""

    genome_id: str
    path: Path
    seq_type: str  # "nucleotide" | "protein"
    n_sequences: int
    n_residues: int
    sha256: str
    n_bytes: int

    @property
    def is_nucleotide(self) -> bool:
        return self.seq_type == "nucleotide"


def _strip_known_extensions(name: str, extensions: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Return (genome_id, matched_extension) or (None, None) if no ext matches.

    A trailing ``.gz`` is peeled first, then the bioinformatics extension. This
    is intentionally case-insensitive on the extension but preserves the case of
    the genome ID (accession IDs can be case-significant).
    """
    stem = name
    if stem.lower().endswith(".gz"):
        stem = stem[: -len(".gz")]
    lower = stem.lower()
    for ext in extensions:
        if lower.endswith(ext.lower()):
            return stem[: -len(ext)], ext
    return None, None


def _sha256(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def discover_genomes(cfg: Config) -> list[Genome]:
    """Discover and validate all genome FASTA files under ``cfg.input_dir``.

    Nucleotide and (optionally) protein files are considered based on their
    extension; the actual sequence type is confirmed by inspecting content, and
    a mismatch (e.g. a ``.fna`` that is actually protein) is reported via the
    ``seq_type`` field so the annotation stage can route or skip it.
    """
    in_dir = Path(cfg.input_dir)
    considered_exts = tuple(cfg.nucleotide_extensions) + tuple(cfg.protein_extensions)

    # Sort for deterministic processing order.
    candidates = sorted(
        (p for p in in_dir.iterdir() if p.is_file()),
        key=lambda p: p.name,
    )

    genomes: dict[str, Genome] = {}
    errors: list[str] = []

    for path in candidates:
        genome_id, matched_ext = _strip_known_extensions(path.name, considered_exts)
        if genome_id is None:
            continue  # not a recognized FASTA extension; ignore quietly
        if not genome_id:
            errors.append(f"{path.name}: filename has no genome ID before the extension")
            continue

        try:
            stats: FastaStats = validate_fasta(path)
        except FastaError as exc:
            errors.append(str(exc))
            continue

        sha, n_bytes = _sha256(path)

        if genome_id in genomes:
            errors.append(
                f"Duplicate genome ID '{genome_id}' from '{path.name}' "
                f"(already seen as '{Path(genomes[genome_id].path).name}')"
            )
            continue

        genomes[genome_id] = Genome(
            genome_id=genome_id,
            path=path,
            seq_type=stats.seq_type,
            n_sequences=stats.n_sequences,
            n_residues=stats.n_residues,
            sha256=sha,
            n_bytes=n_bytes,
        )

    if errors:
        joined = "\n  - ".join(errors)
        raise DiscoveryError(
            f"Discovery found {len(errors)} problem(s) in {in_dir}:\n  - {joined}"
        )

    if not genomes:
        raise DiscoveryError(
            f"No FASTA files found in {in_dir} matching extensions "
            f"{list(considered_exts)}"
        )

    return [genomes[k] for k in sorted(genomes)]
