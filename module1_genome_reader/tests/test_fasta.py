from __future__ import annotations

import gzip

import pytest

from genome_reader.fasta import FastaError, validate_fasta


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_detects_nucleotide(tmp_path):
    p = _write(tmp_path / "g.fna", ">c1\nACGTACGTNNNACGTACGT\n")
    stats = validate_fasta(p)
    assert stats.seq_type == "nucleotide"
    assert stats.n_sequences == 1


def test_detects_protein(tmp_path):
    p = _write(tmp_path / "g.faa", ">p1\nMEEILPQFWYMEEILPQFWY\n")
    assert validate_fasta(p).seq_type == "protein"


def test_multi_record_counts(tmp_path):
    p = _write(tmp_path / "g.fna", ">c1\nACGT\nACGT\n>c2\nTTTT\n")
    stats = validate_fasta(p)
    assert stats.n_sequences == 2
    assert stats.n_residues == 12


def test_empty_file_raises(tmp_path):
    p = _write(tmp_path / "g.fna", "")
    with pytest.raises(FastaError):
        validate_fasta(p)


def test_no_header_raises(tmp_path):
    p = _write(tmp_path / "g.fna", "ACGTACGT\n")
    with pytest.raises(FastaError):
        validate_fasta(p)


def test_header_only_raises(tmp_path):
    p = _write(tmp_path / "g.fna", ">c1\n")
    with pytest.raises(FastaError):
        validate_fasta(p)


def test_gzip_supported(tmp_path):
    p = tmp_path / "g.fna.gz"
    with gzip.open(p, "wb") as fh:
        fh.write(b">c1\nACGTACGTACGT\n")
    stats = validate_fasta(p)
    assert stats.is_gzipped
    assert stats.seq_type == "nucleotide"
