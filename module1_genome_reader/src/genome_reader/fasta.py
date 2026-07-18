"""Minimal, dependency-free FASTA reading and validation.

We deliberately avoid Biopython here: the only things Module 1 needs from a
FASTA file are (a) is it parseable and non-empty, and (b) is it nucleotide or
protein. A tiny purpose-built reader keeps the dependency surface small and the
behavior easy to test deterministically. Gzip-compressed inputs are handled
transparently.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Characters that count as "nucleotide" for the seq-type heuristic: the four DNA
# bases plus U (RNA) and N (any). We deliberately EXCLUDE the IUPAC ambiguity
# codes (R,Y,S,W,K,M,B,D,H,V) here -- every one of them is also a valid amino
# acid letter, so counting them would inflate the nucleotide score of a protein
# and misclassify it. Assemblies contain far less than 10% ambiguity codes, so
# the >=90% threshold still classifies real DNA correctly.
_NUCLEOTIDE_ALPHABET = frozenset("ACGTUN")


class FastaError(ValueError):
    """Raised when a FASTA file is malformed, empty, or unreadable."""


@dataclass(frozen=True)
class FastaStats:
    """Summary of a validated FASTA file."""

    path: str
    n_sequences: int
    n_residues: int
    seq_type: str  # "nucleotide" | "protein"
    is_gzipped: bool


def _open_text(path: Path) -> io.TextIOBase:
    """Open a possibly-gzipped file as text."""
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_records(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(header, sequence)`` tuples from a FASTA file.

    ``header`` excludes the leading ``>``. Raises :class:`FastaError` if the
    file does not start with a header line.
    """
    with _open_text(path) as handle:
        header: str | None = None
        chunks: list[str] = []
        first = True
        for raw in handle:
            line = raw.rstrip("\n").rstrip("\r")
            if first:
                first = False
                if not line.startswith(">"):
                    raise FastaError(
                        f"{path}: does not start with a FASTA header ('>')"
                    )
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks)


def validate_fasta(path: Path) -> FastaStats:
    """Validate a FASTA file and classify it as nucleotide or protein.

    Fails loudly (``FastaError``) when the file is unreadable, has no records,
    or contains a record with no sequence characters. Sequence-type detection
    uses the fraction of residues drawn from the nucleotide alphabet: >= 90%
    nucleotide characters classifies the file as nucleotide, otherwise protein.
    """
    if not path.exists():
        raise FastaError(f"{path}: file does not exist")
    if path.stat().st_size == 0:
        raise FastaError(f"{path}: file is empty (0 bytes)")

    n_sequences = 0
    n_residues = 0
    n_nucleotide = 0
    try:
        for header, seq in iter_records(path):
            n_sequences += 1
            seq_upper = seq.upper()
            for ch in seq_upper:
                if ch in ("-", "*", ".", " "):
                    continue  # gaps / stops don't inform the alphabet vote
                n_residues += 1
                if ch in _NUCLEOTIDE_ALPHABET:
                    n_nucleotide += 1
    except FastaError:
        raise
    except (OSError, UnicodeDecodeError, gzip.BadGzipFile) as exc:
        raise FastaError(f"{path}: could not read file: {exc}") from exc

    if n_sequences == 0:
        raise FastaError(f"{path}: no FASTA records found")
    if n_residues == 0:
        raise FastaError(f"{path}: FASTA records contain no sequence characters")

    frac_nuc = n_nucleotide / n_residues
    seq_type = "nucleotide" if frac_nuc >= 0.90 else "protein"

    return FastaStats(
        path=str(path),
        n_sequences=n_sequences,
        n_residues=n_residues,
        seq_type=seq_type,
        is_gzipped=path.suffix == ".gz",
    )
