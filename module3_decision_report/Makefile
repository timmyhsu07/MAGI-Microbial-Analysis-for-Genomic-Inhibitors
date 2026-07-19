# Module 3 (The Decision Report). Everything reproducible from these targets.
#
# The venv is created with uv; if you don't have uv, a plain
# `python -m venv .venv && .venv/bin/pip install -e '.[test]'` works too.

VENV := .venv
PY := $(VENV)/bin/python

.PHONY: venv test demo evaluate app clean

venv:
	uv venv --python 3.11 $(VENV)
	uv pip install --python $(PY) -e '.[test]'

# The acceptance gate for this module.
test:
	$(PY) -m pytest

# Run the crafted demo cases (one per decision branch) and print each report.
demo:
	$(PY) -m decision_report.cli demo

# Run the synthetic held-out evaluation panel.
evaluate:
	$(PY) -m decision_report.cli evaluate

# Run the Streamlit decision-report interface.
app:
	$(PY) -m streamlit run src/decision_report/app.py

clean:
	rm -rf out .pytest_cache
	find src tests -name __pycache__ -type d -prune -exec rm -rf {} +
