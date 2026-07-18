from __future__ import annotations

import pytest

from genome_reader.config import Config
from genome_reader.discovery import DiscoveryError, discover_genomes

NUC = ">c1\nACGTACGTNNNACGTACGT\n"


def _cfg(input_dir, output_dir, **kw):
    return Config(input_dir=str(input_dir), output_dir=str(output_dir), **kw)


def test_extracts_genome_ids_sorted(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "SAMN_02.fna").write_text(NUC)
    (d / "SAMN_01.fna").write_text(NUC)
    genomes = discover_genomes(_cfg(d, tmp_path / "out"))
    assert [g.genome_id for g in genomes] == ["SAMN_01", "SAMN_02"]


def test_strips_gz_and_ext(tmp_path):
    import gzip

    d = tmp_path / "in"
    d.mkdir()
    with gzip.open(d / "acc9.fasta.gz", "wb") as fh:
        fh.write(NUC.encode())
    genomes = discover_genomes(_cfg(d, tmp_path / "out"))
    assert genomes[0].genome_id == "acc9"


def test_duplicate_id_raises(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "acc.fna").write_text(NUC)
    (d / "acc.fasta").write_text(NUC)
    with pytest.raises(DiscoveryError, match="Duplicate genome ID"):
        discover_genomes(_cfg(d, tmp_path / "out"))


def test_protein_detected(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "prot.faa").write_text(">p\nMEEILPQFWYMEEILPQFWY\n")
    genomes = discover_genomes(_cfg(d, tmp_path / "out"))
    assert genomes[0].seq_type == "protein"


def test_malformed_fasta_aborts(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "good.fna").write_text(NUC)
    (d / "bad.fna").write_text("not a fasta at all\n")
    with pytest.raises(DiscoveryError):
        discover_genomes(_cfg(d, tmp_path / "out"))


def test_empty_dir_raises(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    with pytest.raises(DiscoveryError, match="No FASTA files"):
        discover_genomes(_cfg(d, tmp_path / "out"))


def test_checksum_is_stable(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "acc.fna").write_text(NUC)
    g1 = discover_genomes(_cfg(d, tmp_path / "o1"))[0]
    g2 = discover_genomes(_cfg(d, tmp_path / "o2"))[0]
    assert g1.sha256 == g2.sha256
