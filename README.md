# MAGI

**Microbial Analysis for Genomic Inhibitors**

MAGI is a research project for antimicrobial-resistance screening from assembled bacterial genomes. It annotates resistance features, trains one model per antibiotic, and turns each prediction into a report with supporting evidence and explicit no-calls.

> MAGI is decision-support software, not a treatment recommendation. Results must be confirmed with standard laboratory antimicrobial-susceptibility testing.

## Pipeline

| Module | Input | Output |
|---|---|---|
| [Genome Reader](module1_genome_reader/README.md) | Assembled FASTA files | AMRFinderPlus feature matrix, hit table, schema, and run manifest |
| [Predictor](module2_predictor/README.md) | Feature matrix, phenotype labels, and target-gene table | Calibrated per-drug models, metrics, and decision logs |
| [Decision Report](module3_decision_report/README.md) | Module 1 features and Module 2 predictions | Streamlit report cards, evidence categories, no-calls, and evaluation tables |

The modules communicate through saved artifacts. Module 2 uses the feature order written by Module 1, and Module 3 reads the model artifacts produced by Module 2. The UI supports both the bundled demonstration data and trained real-pipeline artifacts.

## Run the web app

Python 3.11 and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
cd module3_decision_report
make venv
make app
```

The app starts in demonstration mode. Select **Real pipeline** in the sidebar to load existing Module 1 output and Module 2 models.

## Run the integrated demo

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e module2_predictor -e 'module3_decision_report[test]'
.venv/bin/python scripts/demo_real_pipeline.py
```

The demo trains Module 2 on its fixture corpus, loads the resulting models through `decision_report.real_pipeline`, and generates Module 3 decisions for several genomes. It does not require AMRFinderPlus because the fixture feature matrix is already included.

## Decision logic

- One class-balanced logistic-regression model is trained per antibiotic.
- Genetic clusters stay together during cross-validation to reduce train/test leakage.
- Isotonic calibration is fit on a separate calibration split.
- Target-gene absence, out-of-distribution inputs, and low-confidence probabilities can override the model or produce a no-call.
- Reports distinguish known resistance mechanisms from statistical associations and missing resistance signals.

The bundled trained example covers *Escherichia coli* and three antibiotics: ciprofloxacin, ampicillin, and gentamicin. The mock UI includes extra drugs so each report state can be demonstrated.

## Current limitations

- The Streamlit app opens in demonstration mode; its synthetic values are not clinical results.
- Module 1 does not generate the target-gene presence table used by the deterministic gate. Real runs must supply that table separately.
- Jaccard distance over AMR features is used as an offline proxy for whole-genome Mash distance.
- During cross-validation, the OOD reference pool can include other genomes from the same test fold. This can make the reported OOD behavior optimistic, but it does not change the saved deployment models.
- The pipeline begins with assembled, quality-controlled FASTA files. Read assembly, species identification, and metagenomic binning are out of scope.

## Repository guide

| Path | Purpose |
|---|---|
| [`module1_genome_reader/`](module1_genome_reader/README.md) | FASTA annotation and feature generation |
| [`module2_predictor/`](module2_predictor/README.md) | Model training, calibration, gating, and evaluation |
| [`module3_decision_report/`](module3_decision_report/README.md) | Decision rules, evidence rendering, CLI, and Streamlit app |
| [`scripts/demo_real_pipeline.py`](scripts/demo_real_pipeline.py) | End-to-end command-line demo |
| [`RUN_150.md`](RUN_150.md) | Reproduction steps for a larger *E. coli* cohort |
| [`DEPLOY.md`](DEPLOY.md) | Streamlit deployment notes |
