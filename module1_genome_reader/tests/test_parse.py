from __future__ import annotations

import pytest

from genome_reader import constants
from genome_reader.parse import ParseError, feature_kind_for, parse_tsv

HEADER_4X = (
    "Protein id\tContig id\tStart\tStop\tStrand\tElement symbol\tElement name\t"
    "Scope\tType\tSubtype\tClass\tSubclass\tMethod\tTarget length\t"
    "Reference sequence length\t% Coverage of reference\t% Identity to reference\t"
    "Alignment length\tClosest reference accession\tClosest reference name\t"
    "HMM accession\tHMM description"
)

# Legacy 3.x header spellings, to prove the alias table works.
HEADER_3X = (
    "Protein identifier\tContig id\tStart\tStop\tStrand\tGene symbol\tSequence name\t"
    "Scope\tElement type\tElement subtype\tClass\tSubclass\tMethod\tTarget length\t"
    "Reference sequence length\t% Coverage of reference sequence\t"
    "% Identity to reference sequence\tAlignment length\tAccession of closest sequence\t"
    "Name of closest sequence\tHMM id\tHMM description"
)

ROW = (
    "NA\tcontig1\t100\t960\t+\tblaTEM-1\tbeta-lactamase\tcore\tAMR\tAMR\t"
    "BETA-LACTAM\tBETA-LACTAM\tEXACTX\t286\t286\t100.00\t99.65\t286\t"
    "NG_050145.1\tblaTEM-1\tNA\tNA"
)


def _write(path, header, *rows):
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_parse_4x(tmp_path):
    p = _write(tmp_path / "g.tsv", HEADER_4X, ROW)
    recs = parse_tsv(p, "g")
    assert len(recs) == 1
    r = recs[0]
    assert r["genome_id"] == "g"
    assert r[constants.FIELD_ELEMENT_SYMBOL] == "blaTEM-1"
    assert r[constants.FIELD_ELEMENT_TYPE] == "AMR"
    assert r[constants.FIELD_PCT_IDENTITY] == 99.65
    assert r[constants.FIELD_PCT_COVERAGE] == 100.00
    assert r["feature_kind"] == constants.FEATURE_KIND_ACQUIRED


def test_parse_legacy_3x_aliases(tmp_path):
    p = _write(tmp_path / "g.tsv", HEADER_3X, ROW)
    recs = parse_tsv(p, "g")
    assert recs[0][constants.FIELD_ELEMENT_SYMBOL] == "blaTEM-1"
    assert recs[0][constants.FIELD_PCT_IDENTITY] == 99.65


def test_point_mutation_feature_kind(tmp_path):
    point_row = ROW.replace("\tAMR\tAMR\t", "\tAMR\tPOINT\t").replace(
        "blaTEM-1\tbeta-lactamase", "gyrA_S83L\tgyrA point"
    )
    p = _write(tmp_path / "g.tsv", HEADER_4X, point_row)
    recs = parse_tsv(p, "g")
    assert recs[0]["feature_kind"] == constants.FEATURE_KIND_POINT


def test_zero_hits_returns_empty(tmp_path):
    p = _write(tmp_path / "g.tsv", HEADER_4X)
    assert parse_tsv(p, "g") == []


def test_missing_required_column_raises(tmp_path):
    bad_header = HEADER_4X.replace("Element symbol", "Something else")
    p = _write(tmp_path / "g.tsv", bad_header, ROW)
    with pytest.raises(ParseError, match="missing required column"):
        parse_tsv(p, "g")


def test_ragged_row_raises(tmp_path):
    p = _write(tmp_path / "g.tsv", HEADER_4X, "NA\tcontig1\t100")
    with pytest.raises(ParseError, match="expected"):
        parse_tsv(p, "g")


def test_feature_kind_for():
    assert feature_kind_for("POINT") == constants.FEATURE_KIND_POINT
    assert feature_kind_for("point") == constants.FEATURE_KIND_POINT
    assert feature_kind_for("AMR") == constants.FEATURE_KIND_ACQUIRED
    assert feature_kind_for(None) == constants.FEATURE_KIND_ACQUIRED
