# Module 2 (Predictor). Everything reproducible from these targets.
#
# The venv is created with uv; if you don't have uv, a plain
# `python -m venv .venv && .venv/bin/pip install -e '.[test]'` works too.

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv fixtures test train clean

venv:
	uv venv --python 3.11 $(VENV)
	uv pip install --python $(PY) -e '.[test]'

# Regenerate the synthetic fixture corpus (deterministic; pure function of SEED).
fixtures:
	$(PY) scripts/make_fixtures.py

# The acceptance gate for this module.
test:
	$(PY) -m pytest

# Train per-drug models + metrics from the reference config onto out/.
train:
	$(PY) -m predictor.cli --config contracts/config.yaml

clean:
	rm -rf out .pytest_cache
	find src tests -name __pycache__ -type d -prune -exec rm -rf {} +
