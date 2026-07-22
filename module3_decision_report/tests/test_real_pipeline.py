"""Integration tests for the Module 1 -> Module 2 -> Module 3 artifact path.

Module 2 is trained on its fixture corpus, then the resulting artifacts are
loaded through ``real_pipeline`` and passed to ``report.build_report``. The
fixture corpus follows the Module 1 output schema, so AMRFinderPlus is not
required for this test.

Skipped automatically if the ``predictor`` package (Module 2) is not installed
alongside ``decision_report`` in this environment.
"""

from __future__ import annotations

import pathlib

import pytest

predictor_config = pytest.importorskip("predictor.config")
predictor_train = pytest.importorskip("predictor.train")

from decision_report.contracts import DecisionLabel
from decision_report.real_pipeline import IntegrationError, Module1FeatureStore, ModelPredictor
from decision_report.report import build_report
from decision_report.config import DecisionConfig

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE2_FIXTURES = REPO_ROOT / "module2_predictor" / "tests" / "fixtures"
MODULE2_CONFIG = REPO_ROOT / "module2_predictor" / "contracts" / "config.yaml"

SPECIES = "Escherichia coli"


@pytest.fixture(scope="module")
def trained_models_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("module2_real_train")
    cfg = predictor_config.load_config(MODULE2_CONFIG, overrides={"output_dir": str(out)})
    predictor_train.run(cfg)
    return out / "models"


def test_module1_feature_store_reconstructs_hits_from_schema_without_long_table():
    store = Module1FeatureStore(MODULE2_FIXTURES)
    assert store.used_reconstructed_hits is True

    bundle = store("STR01_ISO01")
    assert bundle.genome_id == "STR01_ISO01"
    assert bundle.binary_row["blaTEM-1"] == 1
    assert any(h.element_symbol == "blaTEM-1" for h in bundle.long_hits)


def test_module1_feature_store_accepts_a_fasta_like_path():
    store = Module1FeatureStore(MODULE2_FIXTURES)
    bundle = store("/some/dir/STR01_ISO01.fasta.gz")
    assert bundle.genome_id == "STR01_ISO01"


def test_module1_feature_store_raises_for_unknown_genome():
    store = Module1FeatureStore(MODULE2_FIXTURES)
    with pytest.raises(IntegrationError, match="not found"):
        store("no_such_genome")


def test_model_predictor_covers_exactly_the_trained_drugs(trained_models_dir):
    predictor = ModelPredictor(
        trained_models_dir, MODULE2_FIXTURES / "target_genes.csv", species=SPECIES
    )
    assert predictor.covered_drugs() == ["ampicillin", "ciprofloxacin", "gentamicin"]
    assert predictor.covered_species() == [SPECIES]
    assert predictor.ood_threshold() == pytest.approx(0.5)


def test_real_pipeline_end_to_end_produces_a_valid_report(trained_models_dir):
    store = Module1FeatureStore(MODULE2_FIXTURES)
    predictor = ModelPredictor(
        trained_models_dir, MODULE2_FIXTURES / "target_genes.csv", species=SPECIES
    )
    config = DecisionConfig(
        covered_species=(SPECIES,), ood_threshold=predictor.ood_threshold()
    )

    features = store("STR04_ISO02")  # make_fixtures.py knocks out both cipro targets here
    report = build_report(features, predictor, config, species=SPECIES)

    assert report.species_supported is True
    assert report.covered_drugs == ["ampicillin", "ciprofloxacin", "gentamicin"]
    assert {d.drug for d in report.decisions} == set(report.covered_drugs)
    assert "DECISION SUPPORT ONLY" in report.disclaimer

    cipro = next(d for d in report.decisions if d.drug == "ciprofloxacin")
    assert cipro.label is DecisionLabel.LIKELY_TO_FAIL
    assert cipro.intrinsic_resistance is True  # target gate fired: both gyrA and parC absent

    for decision in report.decisions:
        assert isinstance(decision.label, DecisionLabel)
