# MAGI

**Microbial Analysis for Genomic Inhibitors**

A **read-only, defensive** antimicrobial-resistance (AMR) decision-support pipeline: given an already-assembled, QC'd bacterial genome FASTA, predict which of a small set of antibiotics are likely to work, and show the evidence behind every call. No module in this repo generates, designs, mutates, or optimizes a sequence or organism — every stage only reads and reasons about resistance determinants that already exist.

> Historical Phase 1 requirements audit: **[`AUDIT.md`](AUDIT.md)**. The current implementation and limitations are summarized below.

## The three modules, and how they actually connect

```
  assembled FASTA                    features_binary.parquet          per-drug .joblib
  (QC'd, one per                     features_long.parquet            + .model.json
   genome)                           feature_schema.json              (frozen columns,
       |                                    |                          weights, isotonic
       v                                    v                          calibrator, target
  +--------------+   AMRFinderPlus   +--------------+   grouped CV,   gate, OOD refs)
  |   Module 1   |------------------>|   Module 2   |---------------->|
  | Genome Reader|                   |  Predictor   |   calibration   |
  +--------------+                   +--------------+                |
                                            |                          v
                              target_genes.csv (separate,        +--------------+
                              hand-curated input --              |   Module 3   |
                              see "Known gaps" below)  --------->| Decision     |
                                                                  | Report       |
                                                                  +--------------+
                                                                        |
                                                                        v
                                                        per-drug card: likely to fail /
                                                        likely to work / no-call, evidence
                                                        category, supporting genes, SHAP
                                                        caveat, mandatory lab-testing banner
```

Each module is an independently installable Python package (`module1_genome_reader`, `module2_predictor`, `module3_decision_report`), each with its own `pyproject.toml`, tests, and README. **They are wired together by a shared artifact contract, not by importing each other's internals**:

- Module 1's contract with Module 2 is `feature_schema.json` + `features_binary.parquet` (documented in [`module1_genome_reader/OUTPUT_FORMAT_SPEC.md`](module1_genome_reader/OUTPUT_FORMAT_SPEC.md)). Module 2 freezes that exact column list into every trained model artifact (`module2_predictor/contracts/model_artifact.schema.json`) and never reshapes it at inference time.
- Module 2's contract with Module 3 is the per-drug `.joblib` artifact plus the target-gene table. Module 3 defines this as a `Predictor` Protocol ([`module3_decision_report/src/decision_report/contracts.py`](module3_decision_report/src/decision_report/contracts.py)) so it can run against either a mock (for demos/tests) or the real thing.

### The adapter that makes it real: `decision_report.real_pipeline`

Module 3 ships two things that satisfy its `FeatureExtractor`/`Predictor` protocols:

| | Purpose | Backed by |
|---|---|---|
| `mock_pipeline.py` | Demos, unit tests, "try Module 3 with zero real artifacts" | Seeded, deterministic, clearly-labeled fake data |
| `real_pipeline.py` | The actual pipeline | A real Module 1 output directory + real Module 2 `.joblib` artifacts |

`real_pipeline.py` (added to close the integration gap `AUDIT.md` flagged: *"Modules 2 and 3 have never been run together"*) provides:

- **`Module1FeatureStore`** — loads a real Module 1 `output_dir` once and serves `FeatureBundle` lookups by genome ID (or FASTA path). Uses the real per-hit `features_long.parquet` when present; falls back to reconstructing hits from `feature_schema.json` metadata when a directory only has the binary matrix (this is exactly how Module 2's own fixture corpus is shaped, since it never runs AMRFinderPlus). The fallback is flagged via `.used_reconstructed_hits`, never silently passed off as real provenance.
- **`ModelPredictor`** — loads Module 2's real `.joblib` artifacts and implements the `Predictor` protocol. Calibrated probability and the target-gate call are read via `predictor.inference.predict_one_genome` — **the exact function Module 2 itself uses at inference time**, never recomputed. Feature-importance and a continuous out-of-distribution distance are derived from the stored model weights and `predictor.distance.nearest_jaccard`. Critically, **`covered_drugs()` is derived from whatever models are actually on disk** — not a second hardcoded list — so the covered-drug set can never drift out of sync with what was actually trained (this is what fixed the audit's "drug set declared twice, inconsistently" finding).

Proven end-to-end, not just described: [`module3_decision_report/tests/test_real_pipeline.py`](module3_decision_report/tests/test_real_pipeline.py) trains Module 2 for real on its own fixture corpus and runs the result through Module 3's real decision engine — 5 passing tests, no mocks. Run it yourself:

```bash
pip install -e module2_predictor -e 'module3_decision_report[test]'
python scripts/demo_real_pipeline.py
```

That script trains Module 2, wires the trained artifacts through `real_pipeline` into Module 3, and prints a real per-drug decision card for a handful of genomes — including a genome where the deterministic target gate actually fires (`ciprofloxacin: likely to fail`, because both `gyrA` and `parC` are absent).

## Quickstart per module

```bash
# Module 1 mock run (see its README for a real AMRFinderPlus run)
(cd module1_genome_reader \
  && python3 -m venv .venv \
  && .venv/bin/pip install -e '.[test]' \
  && .venv/bin/python scripts/demo_mock_run.py)

# Module 2 (no external tool needed; trains on its own synthetic fixture corpus)
(cd module2_predictor && make venv && make fixtures && make test && make train)

# MAGI web app (mock pipeline, zero real artifacts needed)
(cd module3_decision_report && make venv && make test && make app)

# The real, interconnected pipeline (from the repo root)
pip install -e module2_predictor -e 'module3_decision_report[test]'
python scripts/demo_real_pipeline.py
```

## What is and is not covered

- **Species**: Module 1's AMRFinderPlus wrapper supports any of AMRFinderPlus's ~29 recognized organisms (`--organism`), but the *trained, evaluated* pipeline in this repo covers exactly **one**: *Escherichia coli* (Module 2's fixtures and Module 3's default `DecisionConfig.covered_species` both declare it). Running Module 1 against a different organism does not automatically make Modules 2/3 valid for it — there is no code-level guard preventing that mismatch today (see Known gaps).
- **Antibiotics**: exactly the drugs Module 2 actually trained models for — derived at runtime from whichever `.joblib` artifacts exist in the models directory (today, on the fixture corpus: **ciprofloxacin, ampicillin, gentamicin**). Module 3's `mock_pipeline.py` demonstrates a different, larger illustrative drug set (including colistin and trimethoprim-sulfamethoxazole) purely for showing off every decision branch — that mock set is never presented as the real pipeline's coverage.
- **Everything upstream of an assembled FASTA** — read assembly, species identification, demultiplexing, metagenomic binning — is explicitly out of scope. The pipeline starts at an assembled, QC'd genome.

## Honest limitations (see `AUDIT.md` for full detail and file:line evidence)

- **The Streamlit UI defaults to demonstration mode.** Real-pipeline results require existing Module 1 output plus trained Module 2 model and target-gene artifacts. The interface labels mock output as demonstration data and keeps it separate from real-pipeline provenance.
- **Module 1 does not capture drug-target presence.** The target-gene table the gate depends on (`target_genes.csv`) is a hand-curated Module 2 input, not something any real annotation step produces. In a real deployment this table would need a genuine source.
- **The de-duplication distance is a documented proxy.** Jaccard distance over the AMR feature matrix stands in for a real Mash whole-genome distance; Mash support is fail-fast (never silently substituted) but not implemented.
- **A cross-validated evaluation metric can leak.** The out-of-distribution reference set used while computing *reported* metrics during cross-validation can include other same-fold test genomes, which may understate how often a genuinely novel genome should be flagged OOD. Does not affect the deployed model artifact, only the honesty of `metrics_report.json`'s numbers. Not yet fixed — see `AUDIT.md`.
- **`on_target_absent`'s domain invariant is now enforced, not just documented.** Module 2's config validator rejects `on_target_absent: susceptible` outright (an absent molecular target can never mean "likely to work") — this was a live footgun the audit caught; it's fixed as of this pass.

## Repo layout

| Path | What |
|---|---|
| [`module1_genome_reader/`](module1_genome_reader/README.md) | FASTA → AMR feature matrix via AMRFinderPlus |
| [`module2_predictor/`](module2_predictor/README.md) | Leakage-safe, calibrated, per-drug resistance models + deterministic target gate |
| [`module3_decision_report/`](module3_decision_report/README.md) | Decision/evidence layer + Streamlit UI and CLI; `mock_pipeline.py` for standalone demos, `real_pipeline.py` for the real thing |
| [`scripts/demo_real_pipeline.py`](scripts/demo_real_pipeline.py) | Reproducible end-to-end demo of the real (non-mock) pipeline |
| [`AUDIT.md`](AUDIT.md) | Historical Phase 1 requirements snapshot, with file:line evidence |
