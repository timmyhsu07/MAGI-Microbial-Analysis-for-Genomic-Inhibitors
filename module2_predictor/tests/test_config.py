"""Config validation for the per-drug target gate policy.

``on_target_absent: susceptible`` is deliberately rejected: an absent
molecular target means the drug cannot act on the organism, so the gate must
never resolve to "likely to work" purely from target absence.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from predictor.config import ConfigError, load_config

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _write_config(path, on_target_absent: str):
    cfg = {
        "seed": 1,
        "output_dir": str(path / "out"),
        "inputs": {
            "feature_matrix": str(FIXTURES / "features_binary.parquet"),
            "feature_schema": str(FIXTURES / "feature_schema.json"),
            "labels": str(FIXTURES / "labels.csv"),
            "target_gene_table": str(FIXTURES / "target_genes.csv"),
        },
        "distance": {"method": "jaccard", "mash_sketch_dir": None, "cluster_threshold": 0.05},
        "cv": {"n_splits": 5, "calibration_fraction": 0.25, "min_per_class": 3},
        "model": {"C": 1.0, "max_iter": 2000},
        "ood": {"threshold": 0.5},
        "decision": {"min_confidence": 0.65},
        "drugs": {
            "ampicillin": {"target_genes": ["ftsI"], "on_target_absent": on_target_absent},
        },
    }
    config_path = path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    return config_path


def test_on_target_absent_rejects_susceptible(tmp_path):
    config_path = _write_config(tmp_path, "susceptible")
    with pytest.raises(ConfigError, match="resistant.*no_call|susceptible"):
        load_config(config_path)


@pytest.mark.parametrize("value", ["resistant", "no_call"])
def test_on_target_absent_accepts_safe_values(tmp_path, value):
    config_path = _write_config(tmp_path, value)
    cfg = load_config(config_path)
    assert cfg.drugs["ampicillin"]["on_target_absent"] == value
