"""Tests for report assembly's graceful-failure surface (report.build_report /
report.report_from_fasta): unsupported species, unavailable predictor,
uncovered drugs, empty features, and failed feature extraction each yield a
clear message -- never a silent guess.
"""

from __future__ import annotations

import dataclasses

from decision_report.contracts import (
    DecisionLabel,
    FeatureBundle,
    NoCallReason,
    PredictorUnavailable,
)
from decision_report.mock_pipeline import MOCK_SPECIES, MockFeatureExtractor, MockPredictor
from decision_report.report import build_report, report_from_fasta


class _UnavailablePredictor:
    def covered_drugs(self):
        return ["Ciprofloxacin"]

    def covered_species(self):
        return [MOCK_SPECIES]

    def predict(self, features):
        raise PredictorUnavailable("artifact not found")


def test_unsupported_species_yields_no_decisions(config):
    features = FeatureBundle("g1", {}, [])
    report = build_report(features, MockPredictor(), config, species="Klebsiella pneumoniae")
    assert report.species_supported is False
    assert report.decisions == []
    assert report.errors


def test_predictor_unavailable_yields_no_decisions(config):
    features = FeatureBundle("g2", {}, [])
    report = build_report(features, _UnavailablePredictor(), config, species=MOCK_SPECIES)
    assert report.species_supported is True
    assert report.decisions == []
    assert any("unavailable" in e.lower() for e in report.errors)


def test_uncovered_drug_of_interest_is_a_no_call(config):
    cfg = dataclasses.replace(
        config, drugs_of_interest=(*MockPredictor().covered_drugs(), "Meropenem")
    )
    features = FeatureBundle("g3", {}, [])
    report = build_report(features, MockPredictor(), cfg, species=MOCK_SPECIES)
    assert "Meropenem" in report.uncovered_drugs_requested
    mero = next(d for d in report.decisions if d.drug == "Meropenem")
    assert mero.label is DecisionLabel.NO_CALL
    assert mero.no_call_reason is NoCallReason.DRUG_NOT_COVERED


def test_empty_features_are_flagged(config):
    features = FeatureBundle("g4", {}, [])
    report = build_report(features, MockPredictor(), config, species=MOCK_SPECIES)
    assert any("no features were extracted" in e.lower() for e in report.errors)
    assert len(report.decisions) == len(MockPredictor().covered_drugs())


def test_report_from_fasta_reports_extraction_failure(config, tmp_path):
    missing = tmp_path / "does_not_exist.fasta"
    report = report_from_fasta(str(missing), MockFeatureExtractor(), MockPredictor(), config, species=MOCK_SPECIES)
    assert report.decisions == []
    assert any("feature extraction failed" in e.lower() for e in report.errors)


def test_report_from_fasta_succeeds_for_real_file(config, tmp_path):
    fasta = tmp_path / "genome.fasta"
    fasta.write_text(">contig1\nACGT\n")
    report = report_from_fasta(str(fasta), MockFeatureExtractor(), MockPredictor(), config, species=MOCK_SPECIES)
    assert report.species_supported is True
    assert len(report.decisions) == len(MockPredictor().covered_drugs())
