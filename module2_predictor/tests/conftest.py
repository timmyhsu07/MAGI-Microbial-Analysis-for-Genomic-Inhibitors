from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
REFERENCE_CONFIG = ROOT / "contracts" / "config.yaml"


@pytest.fixture
def fixtures_dir() -> pathlib.Path:
    return FIXTURES


@pytest.fixture
def trained_run(tmp_path):
    """Train every drug on the fixture corpus into a temp dir, once per test.

    Returns the run result object; artifacts live under ``tmp_path``.
    """
    from predictor.config import load_config
    from predictor.train import run

    cfg = load_config(REFERENCE_CONFIG, overrides={"output_dir": str(tmp_path), "seed": 7})
    result = run(cfg)
    return result, pathlib.Path(tmp_path)
