#!/usr/bin/env python3
"""Run the three-module pipeline with the bundled fixture feature matrix.

The script trains Module 2, loads its artifacts through
``decision_report.real_pipeline``, and generates Module 3 decisions for several
genomes. AMRFinderPlus is not required because the feature matrix is included.

Requires both packages installed in the current environment:
    pip install -e module2_predictor -e 'module3_decision_report[test]'

Run:
    python scripts/demo_real_pipeline.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE2_FIXTURES = REPO_ROOT / "module2_predictor" / "tests" / "fixtures"
MODULE2_CONFIG = REPO_ROOT / "module2_predictor" / "contracts" / "config.yaml"
SPECIES = "Escherichia coli"
DEMO_GENOMES = ["STR01_ISO01", "STR04_ISO01", "STR04_ISO02", "STR08_ISO01"]


def main() -> int:
    try:
        from predictor.config import load_config
        from predictor.train import run as train_run
    except ImportError:
        print(
            "ERROR: the 'predictor' package (Module 2) is not installed.\n"
            "  pip install -e module2_predictor -e 'module3_decision_report[test]'",
            file=sys.stderr,
        )
        return 1

    from decision_report.config import DecisionConfig
    from decision_report.real_pipeline import Module1FeatureStore, ModelPredictor
    from decision_report.report import build_report

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = pathlib.Path(tmp)
        print(f"Training Module 2 on its fixture corpus -> {out_dir}")
        cfg = load_config(MODULE2_CONFIG, overrides={"output_dir": str(out_dir)})
        result = train_run(cfg)
        print(
            f"  trained {len(result.trained_drugs)} drug model(s) over "
            f"{result.n_genomes} genomes / {result.n_clusters} genetic clusters: "
            f"{', '.join(result.trained_drugs)}\n"
        )

        store = Module1FeatureStore(MODULE2_FIXTURES)
        predictor = ModelPredictor(
            out_dir / "models", MODULE2_FIXTURES / "target_genes.csv", species=SPECIES
        )
        config = DecisionConfig(
            covered_species=(SPECIES,), ood_threshold=predictor.ood_threshold()
        )
        print(f"Loaded real models for: {', '.join(predictor.covered_drugs())}")
        print(f"Feature provenance: {'reconstructed from schema (no features_long.parquet in these fixtures)' if store.used_reconstructed_hits else 'real per-hit long table'}\n")

        for genome_id in DEMO_GENOMES:
            features = store(genome_id)
            report = build_report(features, predictor, config, species=SPECIES)
            print(f"=== {report.genome_id} ===")
            for dec in report.decisions:
                reason = f" [{dec.no_call_reason.value}]" if dec.no_call_reason else ""
                print(
                    f"  {dec.drug}: {dec.label.value}{reason} "
                    f"(p_resistant={dec.calibrated_prob_resistant:.2f}, "
                    f"evidence={dec.evidence_category.value})"
                )
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
