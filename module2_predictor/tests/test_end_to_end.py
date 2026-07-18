"""End-to-end: train on the fixture corpus and check the emitted artifacts.

Covers the module's output contract — a model artifact, reliability plot, and
metrics (overall AND per genetic group) per drug, a per-genome decision log that
separates the model call from the gate/OOD overrides, and the guarantee that at
least one genome comes back as a no-call.
"""

from __future__ import annotations

import json

import pandas as pd

DRUGS = ["ampicillin", "ciprofloxacin", "gentamicin"]
OVERALL_METRICS = {
    "balanced_accuracy",
    "resistant_recall",
    "susceptible_recall",
    "f1",
    "auroc",
    "pr_auc",
    "brier",
}


def test_per_drug_artifacts_exist(trained_run):
    _, out = trained_run
    for drug in DRUGS:
        assert (out / "models" / f"{drug}.model.json").is_file()
        assert (out / "models" / f"{drug}.joblib").is_file()
        assert (out / "reliability" / f"{drug}.png").is_file()
    assert (out / "metrics_report.json").is_file()
    assert (out / "run_manifest.json").is_file()
    assert (out / "decisions.csv").is_file()


def test_model_artifact_shape(trained_run):
    _, out = trained_run
    art = json.loads((out / "models" / "ciprofloxacin.model.json").read_text())
    assert art["drug"] == "ciprofloxacin"
    assert art["schema_version"] == "1.0.0"
    # One coefficient per frozen feature column, in the same order.
    assert len(art["model"]["coef"]) == len(art["feature_columns"])
    assert art["calibrator"]["kind"] == "isotonic"
    assert art["gate"]["target_genes"] == ["gyrA", "parC"]


def test_metrics_report_has_overall_and_per_group(trained_run):
    _, out = trained_run
    report = json.loads((out / "metrics_report.json").read_text())
    for drug in DRUGS:
        entry = report["drugs"][drug]
        assert OVERALL_METRICS.issubset(entry["overall"])
        # resistant/susceptible recall are reported separately, never merged.
        assert entry["overall"]["resistant_recall"] is not None
        assert entry["overall"]["susceptible_recall"] is not None
        # metrics are broken down per genetic group as well.
        assert len(entry["per_group"]) >= 1


def test_decision_log_separates_model_and_gate(trained_run):
    _, out = trained_run
    dec = pd.read_csv(out / "decisions.csv")

    # Every override is auditable: the raw model call and the final call are both
    # present, along with what forced the final call.
    for col in ("genome_id", "drug", "model_call", "final_call", "decision_source", "calibrated_prob"):
        assert col in dec.columns

    # The gate fires somewhere (fixtures knock out a target for each drug), and a
    # gate firing is logged as its own decision source, distinct from the model.
    assert (dec["decision_source"] == "target_gate").any()

    # Acceptance: at least one genome is a no-call.
    assert (dec["final_call"] == "no_call").any()
