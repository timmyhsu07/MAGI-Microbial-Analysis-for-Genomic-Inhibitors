"""End-to-end pipeline tests using the mock AMRFinderPlus runner.

Covers the assertions required by the module spec: matrix shape, determinism
(run twice -> identical), all-zero-row handling, and schema validity.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from conftest import FROZEN_TIME, MockAmrfinderRunner, frozen_clock
from genome_reader import constants
from genome_reader.config import Config
from genome_reader.pipeline import PipelineError, run_pipeline


def _cfg(genome_dir, out, **kw):
    return Config(
        input_dir=str(genome_dir),
        output_dir=str(out),
        organism="Escherichia",
        workers=2,
        **kw,
    )


def _run(genome_dir, out, runner):
    return run_pipeline(
        _cfg(genome_dir, out),
        runner=runner,
        require_amrfinder=False,
        now_fn=frozen_clock,
    )


def test_shape_and_all_zero_row(genome_dir, tmp_path, mock_runner):
    out = tmp_path / "out"
    res = _run(genome_dir, out, mock_runner)
    assert res.n_genomes_in_matrix == 3
    assert res.n_features == 3

    df = pd.read_parquet(out / constants.OUT_BINARY_PARQUET)
    assert list(df[constants.ID_COLUMN]) == ["genome_A", "genome_B", "genome_C"]
    feats = [c for c in df.columns if c != constants.ID_COLUMN]
    assert feats == ["aph(3')-Ia", "blaTEM-1", "gyrA_S83L"]

    # genome_C had zero hits -> present as an all-zero row.
    c = df[df[constants.ID_COLUMN] == "genome_C"].iloc[0]
    assert int(c["aph(3')-Ia"]) == 0
    assert int(c["blaTEM-1"]) == 0
    assert int(c["gyrA_S83L"]) == 0

    a = df[df[constants.ID_COLUMN] == "genome_A"].iloc[0]
    assert int(a["blaTEM-1"]) == 1 and int(a["gyrA_S83L"]) == 1


def test_csv_matches_parquet(genome_dir, tmp_path, mock_runner):
    out = tmp_path / "out"
    _run(genome_dir, out, mock_runner)
    from_parquet = pd.read_parquet(out / constants.OUT_BINARY_PARQUET)
    from_csv = pd.read_csv(out / constants.OUT_BINARY_CSV)
    assert list(from_csv.columns) == list(from_parquet.columns)
    assert list(from_csv[constants.ID_COLUMN]) == list(from_parquet[constants.ID_COLUMN])


def test_virulence_in_long_not_in_matrix(genome_dir, tmp_path, mock_runner):
    out = tmp_path / "out"
    _run(genome_dir, out, mock_runner)
    df = pd.read_parquet(out / constants.OUT_BINARY_PARQUET)
    feats = [c for c in df.columns if c != constants.ID_COLUMN]
    assert "stxB" not in feats  # VIRULENCE excluded from binary matrix

    long = pd.read_parquet(out / constants.OUT_LONG_PARQUET)
    assert "stxB" in set(long[constants.FIELD_ELEMENT_SYMBOL])  # retained in provenance
    assert "VIRULENCE" in set(long[constants.FIELD_ELEMENT_TYPE])


def test_schema_validity(genome_dir, tmp_path, mock_runner):
    out = tmp_path / "out"
    _run(genome_dir, out, mock_runner)
    schema = json.loads((out / constants.OUT_SCHEMA_JSON).read_text())
    assert schema["schema_version"] == constants.SCHEMA_VERSION
    assert schema["matrix"]["n_genomes"] == 3
    assert schema["matrix"]["n_features"] == 3
    assert schema["matrix"]["value_encoding"] == "presence_absence"
    assert schema["provenance"]["organism"] == "Escherichia"

    cols = {c["column"]: c for c in schema["columns"]}
    assert set(cols) == {"aph(3')-Ia", "blaTEM-1", "gyrA_S83L"}
    assert cols["gyrA_S83L"]["feature_kind"] == "point_mutation"
    assert cols["blaTEM-1"]["feature_kind"] == "acquired_gene"
    assert "BETA-LACTAM" in cols["blaTEM-1"]["drug_classes"]
    assert cols["blaTEM-1"]["n_genomes_present"] == 2


def test_manifest_records_status(genome_dir, tmp_path, mock_runner):
    out = tmp_path / "out"
    _run(genome_dir, out, mock_runner)
    man = json.loads((out / constants.OUT_MANIFEST_JSON).read_text())
    assert man["counts"]["n_genomes_discovered"] == 4
    assert man["counts"]["n_genomes_in_matrix"] == 3

    status = {i["genome_id"]: i["status"] for i in man["inputs"]}
    included = {i["genome_id"]: i["included_in_matrix"] for i in man["inputs"]}
    assert status["genome_P"] == "skipped_protein"
    assert status["genome_C"] == "annotated"
    assert included["genome_P"] is False
    assert included["genome_A"] is True
    # Provenance versions are recorded (mock -> "unknown", but the key exists).
    assert "amrfinderplus" in man["tools"]
    assert man["tools"]["amrfinderplus"]["software_version"] is not None


def test_determinism_byte_identical(genome_dir, tmp_path):
    o1, o2 = tmp_path / "o1", tmp_path / "o2"
    r1 = _run(genome_dir, o1, MockAmrfinderRunner())
    r2 = _run(genome_dir, o2, MockAmrfinderRunner())

    # Deterministic, input-derived run id.
    assert r1.run_id == r2.run_id

    # Byte-identical primary matrix (CSV) and schema (clock frozen).
    assert (o1 / constants.OUT_BINARY_CSV).read_bytes() == (o2 / constants.OUT_BINARY_CSV).read_bytes()
    assert (o1 / constants.OUT_SCHEMA_JSON).read_bytes() == (o2 / constants.OUT_SCHEMA_JSON).read_bytes()

    # Byte-identical parquet artifacts.
    assert (o1 / constants.OUT_BINARY_PARQUET).read_bytes() == (o2 / constants.OUT_BINARY_PARQUET).read_bytes()
    assert (o1 / constants.OUT_LONG_PARQUET).read_bytes() == (o2 / constants.OUT_LONG_PARQUET).read_bytes()


def test_cache_reused_on_rerun(genome_dir, tmp_path):
    out = tmp_path / "out"
    runner1 = MockAmrfinderRunner()
    _run(genome_dir, out, runner1)
    # First run annotates the 3 nucleotide genomes.
    assert sorted(gid for gid, _ in runner1.calls) == ["genome_A", "genome_B", "genome_C"]

    runner2 = MockAmrfinderRunner()
    _run(genome_dir, out, runner2)
    # Second run hits the cache -> runner not invoked at all.
    assert runner2.calls == []


def test_failure_aborts_unless_allowed(genome_dir, tmp_path):
    class FailingRunner(MockAmrfinderRunner):
        def run(self, genome, out_tsv, mode):
            if genome.genome_id == "genome_B":
                raise RuntimeError("boom")
            super().run(genome, out_tsv, mode)

    with pytest.raises(PipelineError, match="failed annotation"):
        run_pipeline(
            _cfg(genome_dir, tmp_path / "o1"),
            runner=FailingRunner(),
            require_amrfinder=False,
            now_fn=frozen_clock,
        )

    # With allow_partial, the run completes using the genomes that succeeded.
    res = run_pipeline(
        _cfg(genome_dir, tmp_path / "o2", allow_partial=True),
        runner=FailingRunner(),
        require_amrfinder=False,
        now_fn=frozen_clock,
    )
    assert res.n_genomes_in_matrix == 2  # A and C; B failed
