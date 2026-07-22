# Module 2: Predictor

Takes Module 1's binary AMR feature matrix and produces calibrated per-drug
resistance predictions with a deterministic
molecular-target gate on top.

The inputs are a feature matrix, phenotype labels, and a target-gene table. The
outputs are one model artifact per drug, evaluation metrics, and a decision log.
Annotation is handled by Module 1 and presentation by Module 3.

---

## What it does, in order

1. **Freeze the feature vocabulary.** The feature columns come from Module 1's
   `feature_schema.json` and are frozen at train time. Inference maps any new
   genome onto the same columns. Genes unseen during training are dropped and
   logged rather than appended.
2. **De-duplicate by genetic distance.** Near-identical isolates (re-sequenced
   or clonal) are collapsed into clusters so they can't straddle a train/test
   split. Offline this uses Jaccard distance over the AMR matrix; a real run can
   use Mash when the binary and sketches are available.
3. **Grouped cross-validation.** Whole clusters go to train **or** test, never
   both. See [`tests/test_splits_no_leakage.py`](tests/test_splits_no_leakage.py).
4. **One model per drug.** An L2-penalised, class-weight-balanced logistic
   regression, trained only on the fold's fit clusters.
5. **Calibrate probabilities.** Isotonic regression is fit on a separate
   calibration cluster set within each fold, not on the test fold.
6. **Deterministic target gate.** If a drug's molecular target gene is absent
   from a genome, the call is fixed regardless of what the model thinks, and the
   override is logged as its own decision source. `on_target_absent` may only be
   configured as `resistant` or `no_call` — `susceptible` is rejected by config
   validation because target absence cannot imply susceptibility.
7. **OOD guard.** A genome far from everything in the training set is flagged and
   returned as a no-call rather than a confident guess.

Precedence for the final call: **target gate → OOD → low-confidence → model.**
`decisions.csv` records the raw model call and each override separately.

## De-duplication threshold

`distance.cluster_threshold` defaults to **0.05** (single-linkage). At Jaccard
0.05 two genomes share ~95% of their detected determinants, which is where
re-sequenced/clonal isolates sit; distinct strains stay in separate
clusters. It is a **tunable** knob, documented in
[`contracts/config.yaml`](contracts/config.yaml): raise it to be stricter about
independence (fewer, larger groups), lower it to keep more groups. On the small
synthetic fixture corpus the replicate feature-flips push many isolates just
past 0.05, which produces many small clusters in the fixture data.

## Metrics

Reported per drug **and per genetic group** (cluster) in
`metrics_report.json`, all from out-of-fold predictions: balanced accuracy,
**resistant-recall and susceptible-recall separately** (never averaged away),
F1, AUROC, PR-AUC, and Brier score. Metrics that are mathematically undefined on
a tiny slice (e.g. AUROC on a single-class group) are recorded as `null` rather
than smoothed or dropped. A reliability curve per drug lands in `reliability/`.

## Outputs (`output_dir/`)

| File | What |
|------|------|
| `models/<drug>.model.json` | auditable artifact — frozen columns, L2 weights, isotonic knots, gate, thresholds, provenance (see [`contracts/model_artifact.schema.json`](contracts/model_artifact.schema.json)) |
| `models/<drug>.joblib` | the live estimators for inference |
| `reliability/<drug>.png` | calibration reliability curve |
| `metrics_report.json` | per-drug and per-group metrics + skipped drugs |
| `decisions.csv` | per-genome, per-drug log: model call vs gate vs OOD vs final |
| `run_manifest.json` | seed, resolved config, versions, cluster + fold assignments |

## Run it

```bash
make venv        # uv venv + editable install (numpy/pandas/sklearn/scipy/...)
make fixtures    # (re)generate the synthetic fixture corpus — deterministic
make test        # pytest, incl. the no-leakage acceptance test
make train       # train all drugs from contracts/config.yaml onto out/
```

Everything is a pure function of the config + `seed`. Point `inputs.*` in your
own config at real Module 1 artifacts and a real phenotype/target table to train
for real.

## Limitations

- **Read-only decision support, not a verdict.** A statistical association is not
  proof of a causal mechanism; the top contributing features are correlations in
  this corpus, not established biology. Confirm with standard lab susceptibility
  testing.
- **The fixtures are synthetic.** `tests/fixtures/` is a small, deterministic
  stand-in for a real AMRFinderPlus corpus and a real phenotype panel. The
  numbers demonstrate the pipeline, not clinical performance.
- **Jaccard is a proxy for Mash.** The offline distance is coarser than
  whole-genome Mash and is only a fallback for self-contained dev/CI.
- **One species at a time.** Coverage is exactly the drugs present in the labels
  table with enough labelled genomes per class; everything else is reported as
  skipped, not silently guessed.
- **The gate is only as good as the target table.** An unknown target state does
  not fire the gate — it defers to the model and logs the gap.
