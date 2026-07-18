#!/usr/bin/env python3
"""Run the full Module 1 pipeline end-to-end WITHOUT AMRFinderPlus installed.

Uses the bundled mock AMRFinderPlus TSVs (tests/data/mock_amrfinder) via an
injected runner, so you can see every output artifact immediately. This is a
demonstration/smoke aid -- real runs use the CLI (`genome-reader ...`) against a
real AMRFinderPlus install.

    python scripts/demo_mock_run.py [output_dir]
"""

from __future__ import annotations

import pathlib
import shutil
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from genome_reader.config import Config  # noqa: E402
from genome_reader.pipeline import run_pipeline  # noqa: E402

MOCK_DIR = _ROOT / "tests" / "data" / "mock_amrfinder"
NUCLEOTIDE = ">c1 demo contig\nACGTACGTACGTNNNACGTACGTACGT\n"
PROTEIN = ">p1 demo protein\nMEEILPQFWYMEEILPQFWYMEEILPQFWY\n"


class MockRunner:
    """Copies a canned TSV into place instead of running AMRFinderPlus."""

    def run(self, genome, out_tsv, mode):
        src = MOCK_DIR / f"{genome.genome_id}.tsv"
        if not src.exists():
            raise FileNotFoundError(f"no mock TSV for {genome.genome_id}")
        shutil.copyfile(src, out_tsv)


def main() -> int:
    out_root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "example_run"
    genomes = out_root / "genomes"
    genomes.mkdir(parents=True, exist_ok=True)
    for gid in ("genome_A", "genome_B", "genome_C"):
        (genomes / f"{gid}.fna").write_text(NUCLEOTIDE, encoding="utf-8")
    (genomes / "genome_P.faa").write_text(PROTEIN, encoding="utf-8")

    cfg = Config(
        input_dir=str(genomes),
        output_dir=str(out_root / "out"),
        organism="Escherichia",
        workers=2,
    )
    result = run_pipeline(cfg, runner=MockRunner(), require_amrfinder=False)
    print(
        f"run_id={result.run_id}  genomes={result.n_genomes_in_matrix}  "
        f"features={result.n_features}  hits={result.n_hits}"
    )
    print(f"artifacts written under: {out_root / 'out'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
