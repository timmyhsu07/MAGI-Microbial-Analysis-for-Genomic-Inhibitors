from __future__ import annotations

from genome_reader import constants
from genome_reader.config import Config
from genome_reader.matrix import (
    build_binary_matrix,
    build_long_table,
    collect_feature_symbols,
)
from genome_reader.schema import build_schema


def _rec(genome_id, symbol, etype, subtype, **kw):
    r = {
        "genome_id": genome_id,
        constants.FIELD_ELEMENT_SYMBOL: symbol,
        constants.FIELD_ELEMENT_TYPE: etype,
        constants.FIELD_ELEMENT_SUBTYPE: subtype,
        constants.FIELD_CLASS: kw.get("cls"),
        constants.FIELD_SUBCLASS: kw.get("subcls"),
        constants.FIELD_METHOD: kw.get("method", "EXACTX"),
        "feature_kind": (
            constants.FEATURE_KIND_POINT
            if subtype == "POINT"
            else constants.FEATURE_KIND_ACQUIRED
        ),
    }
    return r


def _records():
    return [
        _rec("genome_A", "blaTEM-1", "AMR", "AMR", cls="BETA-LACTAM", subcls="BETA-LACTAM"),
        _rec("genome_A", "gyrA_S83L", "AMR", "POINT", cls="QUINOLONE", method="POINTX"),
        _rec("genome_A", "stxB", "VIRULENCE", "VIRULENCE"),
        _rec("genome_B", "blaTEM-1", "AMR", "AMR", cls="BETA-LACTAM"),
        _rec("genome_B", "aph(3')-Ia", "AMR", "AMR", cls="AMINOGLYCOSIDE", subcls="KANAMYCIN"),
    ]


MATRIX_IDS = ["genome_A", "genome_B", "genome_C"]


def test_collect_symbols_excludes_non_amr():
    syms = collect_feature_symbols(_records(), ("AMR",))
    assert syms == ["aph(3')-Ia", "blaTEM-1", "gyrA_S83L"]  # sorted, stxB excluded


def test_binary_matrix_shape_and_zero_row():
    df = build_binary_matrix(_records(), MATRIX_IDS, ("AMR",))
    assert list(df[constants.ID_COLUMN]) == MATRIX_IDS
    feats = [c for c in df.columns if c != constants.ID_COLUMN]
    assert feats == ["aph(3')-Ia", "blaTEM-1", "gyrA_S83L"]

    a = df[df[constants.ID_COLUMN] == "genome_A"].iloc[0]
    assert int(a["blaTEM-1"]) == 1 and int(a["gyrA_S83L"]) == 1 and int(a["aph(3')-Ia"]) == 0
    b = df[df[constants.ID_COLUMN] == "genome_B"].iloc[0]
    assert int(b["aph(3')-Ia"]) == 1 and int(b["blaTEM-1"]) == 1 and int(b["gyrA_S83L"]) == 0
    c = df[df[constants.ID_COLUMN] == "genome_C"].iloc[0]
    assert int(c["aph(3')-Ia"]) == 0 and int(c["blaTEM-1"]) == 0 and int(c["gyrA_S83L"]) == 0


def test_long_table_retains_all_types():
    df = build_long_table(_records())
    assert list(df.columns) == list(constants.LONG_TABLE_COLUMNS)
    assert len(df) == 5
    assert "stxB" in set(df[constants.FIELD_ELEMENT_SYMBOL])
    assert "VIRULENCE" in set(df[constants.FIELD_ELEMENT_TYPE])


def test_empty_long_table_has_columns():
    df = build_long_table([])
    assert list(df.columns) == list(constants.LONG_TABLE_COLUMNS)
    assert len(df) == 0


def test_schema_annotates_every_column(tmp_path):
    cfg = Config(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        organism="Escherichia",
        include_element_types=("AMR",),
    )
    versions = {
        "amrfinderplus": {
            "software_version": "4.0.19",
            "database_version": "2026-01-15.1",
        }
    }
    feats = collect_feature_symbols(_records(), ("AMR",))
    schema = build_schema(
        _records(), feats, MATRIX_IDS, cfg, versions, generated_at="2026-07-18T00:00:00Z"
    )
    assert schema["schema_version"] == constants.SCHEMA_VERSION
    assert schema["matrix"]["n_features"] == 3
    assert schema["matrix"]["included_element_types"] == ["AMR"]
    assert schema["provenance"]["amrfinderplus_database_version"] == "2026-01-15.1"

    cols = {c["column"]: c for c in schema["columns"]}
    assert set(cols) == set(feats)
    assert cols["gyrA_S83L"]["feature_kind"] == constants.FEATURE_KIND_POINT
    assert cols["blaTEM-1"]["feature_kind"] == constants.FEATURE_KIND_ACQUIRED
    assert "BETA-LACTAM" in cols["blaTEM-1"]["drug_classes"]
    assert cols["blaTEM-1"]["n_genomes_present"] == 2
    assert cols["gyrA_S83L"]["n_genomes_present"] == 1
    assert "POINTX" in cols["gyrA_S83L"]["methods_observed"]
