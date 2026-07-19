# Module 3 — The Decision Report

Turns Module 2's per-drug predictions (plus Module 1's provenance) into a
**human-facing, per-drug decision card**: `likely to fail` / `likely to work` /
`no-call`, an honest evidence category, and a plain-language rationale — never
a silent guess.

> **Defensive scope.** This module only *interprets* predictions Module 2
> already produced. It never makes a treatment decision, never designs,
> modifies, or suggests changes to any organism, and every report ships with a
> mandatory "confirm by laboratory testing" disclaimer.
>
> **Module boundary.** No annotation (Module 1), no modeling/calibration/gating
> (Module 2). Module 3 consumes a `FeatureBundle` (Module 1) and a
> `Predictor.predict(...) -> list[DrugPrediction]` (Module 2) — see
> [`contracts.py`](src/decision_report/contracts.py) for the exact typed
> interfaces — and owns only the decision/evidence/report layer on top. It
> ships mocks satisfying both interfaces
> ([`mock_pipeline.py`](src/decision_report/mock_pipeline.py)) so it runs
> standalone with zero real artifacts; point a real deployment's own Module 1/2
> adapters at [`report.build_report`](src/decision_report/report.py) instead.

---

## The three labels, honestly

| Label | Meaning |
|---|---|
| `likely to fail` | organism predicted **resistant** — the drug likely won't work |
| `likely to work` | organism predicted **susceptible** |
| `no-call` | the evidence doesn't support a confident call — reported, not forced |

A no-call is a **feature**, not a failure of the pipeline: it fires whenever
the evidence is weak, conflicting, or the input is unlike anything in training,
rather than let a model guess past its competence.

## Decision rules (first match wins)

See [`decision.py`](src/decision_report/decision.py) for the implementation;
the module docstring there is the source of truth. In order:

1. **Drug not covered** by the predictor → `NO_CALL(DRUG_NOT_COVERED)`
2. **Probability missing/invalid** → `NO_CALL(INVALID_INPUT)` — refuse to guess
3. **Molecular target absent** → `LIKELY_TO_FAIL`, deterministic, overrides the
   model (a drug can't work against a target the organism doesn't have)
4. **Out-of-distribution** genome (`ood_score >= ood_threshold`) → `NO_CALL`
5. **Uncertainty band** around 0.5 → `NO_CALL`
6. **Known mechanism present but model leans susceptible** → `NO_CALL`
   (conflicting evidence)
7. **No resistance signal but model leans resistant** → `NO_CALL` (conflicting
   evidence)
8. Probability above the band → `LIKELY_TO_FAIL`
9. Probability below the band → `LIKELY_TO_WORK`

Every threshold (`uncertainty_band_low/high`, `ood_threshold`, …) lives in
[`config.py`](src/decision_report/config.py), is documented, and is loadable
from YAML/JSON. Module 3 only *consumes* these values — they must be tuned on a
validation split (Module 2's job), never on the test split this module reports
against.

## Evidence categories

Every decision — including no-calls — is tagged with exactly one honest
evidence category ([`evidence.py`](src/decision_report/evidence.py)):

- **(i) known_resistance_mechanism** — a curated determinant (exact gene /
  allele / point mutation) for this drug's class, or a known-mechanism model
  feature driving the call
- **(ii) statistical_association_only** — driven by non-curated signal (weak
  BLAST/PARTIAL/HMM hits or generic statistical features) — labeled
  *association, not causation*
- **(iii) no_known_resistance_signal** — driven by the *absence* of markers

## Outputs

`GenomeReport` (see [`contracts.py`](src/decision_report/contracts.py)): one
per genome, carrying `species_supported`, a `DrugDecision` per drug
(label, evidence category, rationale, caveats, supporting hits/features), any
uncovered-drug requests, graceful-failure `errors`, and the mandatory
disclaimer. The held-out evaluation panel
([`evaluation.py`](src/decision_report/evaluation.py)) additionally reports,
per drug and per genetic group: AUROC / PR-AUC / Brier / F1, balanced accuracy
with **resistant- and susceptible-recall reported separately**, the no-call
rate alongside accuracy on the remaining called predictions, and a reliability
curve — with genetic groups **unseen in training** broken out so
generalization gaps are visible, not averaged away.

## Run it

```bash
make venv       # uv venv + editable install (numpy/pandas/pyyaml/pytest)
make test       # pytest — the acceptance gate for this module
make demo       # run the 8 crafted demo cases, one per decision branch
make evaluate   # run the synthetic held-out evaluation panel
```

Or directly via the CLI:

```bash
decision-report demo --output-dir out/demo
decision-report evaluate --n 60 --seed 20260718 --output-dir out/eval
decision-report report --fasta path/to/genome.fasta
```

Everything above uses the bundled mocks
([`mock_pipeline.py`](src/decision_report/mock_pipeline.py)) so it runs with
zero real Module 1/2 artifacts. Swap in real adapters satisfying the
`FeatureExtractor`/`Predictor` protocols in `contracts.py` for a real run.

## Wiring it to the real pipeline

Everything above uses `mock_pipeline.py` so the module runs standalone. To run
against **real** Module 1 output and real Module 2 trained artifacts instead,
use `real_pipeline.py`:

```python
from decision_report.real_pipeline import Module1FeatureStore, ModelPredictor
from decision_report.config import DecisionConfig
from decision_report.report import build_report

store = Module1FeatureStore("/path/to/module1/output_dir")
predictor = ModelPredictor(
    "/path/to/module2/out/models", "/path/to/target_genes.csv", species="Escherichia coli"
)
config = DecisionConfig(covered_species=("Escherichia coli",), ood_threshold=predictor.ood_threshold())

features = store("genome_id_or_a_fasta_path")
report = build_report(features, predictor, config, species="Escherichia coli")
```

`Module1FeatureStore` loads a real Module 1 output directory once (binary
matrix + schema, and the real per-hit `features_long.parquet` when present;
otherwise it reconstructs hits from the schema's per-column metadata, flagged
via `.used_reconstructed_hits` so that fallback is never mistaken for real
provenance). `ModelPredictor` loads Module 2's `.joblib` artifacts and derives
`covered_drugs()`/`ood_threshold()` from whatever was actually trained — never
a second hardcoded list. Calibrated probability and the target-gate call come
straight from `predictor.inference.predict_one_genome`, the same function
Module 2 itself uses; Module 3 only adds feature-importance and a continuous
OOD distance so it can own the final decision.

See [`../scripts/demo_real_pipeline.py`](../scripts/demo_real_pipeline.py) for
a runnable end-to-end demo, and
[`tests/test_real_pipeline.py`](tests/test_real_pipeline.py) for the proof
that this actually works (trains Module 2 for real on its own fixtures, then
runs the result through this module's real decision engine).

## Testing

```bash
pip install -e '.[test]'
pytest -v
```

- `test_decision.py` — every rule in the engine, in isolation
- `test_evidence.py` — every evidence-category branch
- `test_report.py` — the graceful-failure surface (unsupported species,
  unavailable predictor, uncovered drugs, empty features, failed extraction)
- `test_evaluation.py` — the evaluation panel's shape and determinism
- `test_end_to_end.py` — all 8 crafted demo cases resolve to the decision
  branch their description claims, through the real orchestration

## Running the app

Install the optional Streamlit dependency and start the decision-report UI:

```bash
pip install -e '.[app]'
make app
```

The app opens with the eight zero-setup mock cases. Real-pipeline mode accepts
paths to existing Module 1 output and Module 2 model/target artifacts; it does
not generate or modify those artifacts.

## Limitations & scope

- **Decision support only.** No label here is a treatment decision; every
  report requires human confirmation by standard laboratory antimicrobial
  susceptibility testing.
- **Association is not causation.** Category (ii) evidence and SHAP/importance
  values reflect statistical correlation in the training corpus, not proven
  biological mechanism.
- **The mocks are synthetic.** `mock_pipeline.py` fabricates plausible,
  seeded, deterministic features/predictions so this module runs standalone —
  the numbers demonstrate the decision logic, not clinical performance.
- **One species at a time**, exactly the species/drugs the connected
  predictor covers; anything else is reported as unsupported/uncovered, never
  silently guessed.
