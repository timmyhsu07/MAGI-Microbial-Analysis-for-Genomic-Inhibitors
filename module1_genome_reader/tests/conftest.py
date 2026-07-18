"""Shared test fixtures.

Makes ``genome_reader`` importable from src without an install, and provides a
mock AMRFinderPlus runner that emits canned TSVs so the whole pipeline runs
without the real tool.
"""

from __future__ import annotations

import pathlib
import shutil
import sys

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

MOCK_TSV_DIR = pathlib.Path(__file__).resolve().parent / "data" / "mock_amrfinder"

# Synthetic sequences chosen so the seq-type heuristic is unambiguous:
# all-core-base nucleotide vs an all-non-core-letter "protein".
NUCLEOTIDE_FASTA = ">c1 test contig\nACGTACGTACGTNNNACGTACGTACGT\n"
PROTEIN_FASTA = ">p1 test protein\nMEEILPQFWYMEEILPQFWYMEEILPQFWY\n"

FROZEN_TIME = "2026-07-18T00:00:00Z"


def frozen_clock() -> str:
    return FROZEN_TIME


class MockAmrfinderRunner:
    """Runner that copies tests/data/mock_amrfinder/<genome_id>.tsv into place.

    A missing fixture raises, so a mismatched genome id fails loudly rather than
    silently producing an empty result.
    """

    def __init__(self, fixture_dir: pathlib.Path = MOCK_TSV_DIR):
        self.fixture_dir = pathlib.Path(fixture_dir)
        self.calls: list[tuple[str, str]] = []

    def run(self, genome, out_tsv, mode):  # matches annotate.Runner protocol
        self.calls.append((genome.genome_id, mode))
        src = self.fixture_dir / f"{genome.genome_id}.tsv"
        if not src.exists():
            raise FileNotFoundError(f"no mock TSV for '{genome.genome_id}': {src}")
        shutil.copyfile(src, out_tsv)


@pytest.fixture
def genome_dir(tmp_path):
    """A directory with three nucleotide genomes and one protein file."""
    d = tmp_path / "genomes"
    d.mkdir()
    for gid in ("genome_A", "genome_B", "genome_C"):
        (d / f"{gid}.fna").write_text(NUCLEOTIDE_FASTA, encoding="utf-8")
    (d / "genome_P.faa").write_text(PROTEIN_FASTA, encoding="utf-8")
    return d


@pytest.fixture
def mock_runner():
    return MockAmrfinderRunner()
