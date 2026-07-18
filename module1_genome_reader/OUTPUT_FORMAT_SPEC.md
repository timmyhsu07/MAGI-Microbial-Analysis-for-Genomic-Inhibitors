# Module 1 - Output Format Specification

**Schema version:** `1.0.0`  ·  **Pipeline version:** `0.1.0`

This document is the contract between Module 1 (The Genome Reader) and any
downstream consumer (Modules 2/3). It defines the artifacts, the feature-matrix
schema, ID conventions, determinism guarantees, and how a new genome maps onto
the fixed column set at inference time.

> Module 1 is a **defensive resistance-detection** component. It annotates and
> tabulates resistance determinants that already exist in an assembled genome.
> It never designs, modifies, or suggests changes to an organism.

---

## 1. Artifacts

All artifacts are written to the configured `output_dir`:

| File | Purpose | Primary? |
|------|---------|----------|
| `features_binary.parquet` | genome × feature presence/absence matrix | **yes** |
| `features_binary.csv` | identical matrix as CSV (byte-stable) | mirror |
| `features_long.parquet` | one row per AMRFinderPlus hit, full metadata | provenance |
| `feature_schema.json` | versioned column manifest | contract |
| `run_manifest.json` | tool/db/dep versions, params, checksums, status | reproducibility |

All artifacts for a run share the same `run_id` (recorded in the manifest).

---

## 2. ID conventions

- **`genome_id`** is derived from each input filename by stripping a trailing
  `.gz` (if present) and then the recognized FASTA extension
  (`.fasta`, `.fa`, `.fna`, `.fas`, `.seq`; proteins `.faa`, `.fpaa`).
  Example: `GCF_000005845.2.fna` → `GCF_000005845.2`;
  `SAMN12345.fasta.gz` → `SAMN12345`.
- The extension match is case-insensitive; the ID itself preserves case.
- Two files mapping to the same `genome_id` is a **hard error** (no silent
  overwrite).
- `genome_id` is the join key across every artifact.

---

## 3. `features_binary` — the primary matrix

- **Rows:** one per genome that produced a valid AMRFinderPlus result
  (status `annotated` or `cached`), ordered ascending by `genome_id`.
  - A genome with **zero hits still appears**, as an all-zero row.
  - Genomes that were **skipped** (protein input) or **failed** annotation are
    *not* rows here; their status is in `run_manifest.json`. An annotation
    failure is therefore never silently encoded as "no resistance genes".
- **Columns:** `genome_id` (string) followed by one column per feature — the
  sorted union of AMRFinderPlus **element symbols** observed across the corpus,
  restricted to the configured `include_element_types` (default `["AMR"]`,
  which covers both acquired AMR genes and AMR point mutations).
- **Column order:** ascending lexicographic by `element_symbol` (byte order).
- **Values:** `0` / `1` presence/absence, dtype `int8`.
- **Point mutations** appear as their own columns; AMRFinderPlus encodes the
  mutation in the symbol itself, e.g. `gyrA_S83L`, `23S_A2059G`.
- If the corpus has zero qualifying hits, the matrix contains only the
  `genome_id` column (all rows, no feature columns).

Example (`features_binary.csv`):

```
genome_id,aph(3')-Ia,blaTEM-1,gyrA_S83L
genome_A,0,1,1
genome_B,1,1,0
genome_C,0,0,0
```

---

## 4. `features_long` — per-hit provenance

One row per AMRFinderPlus hit, retaining **all** element types (AMR, VIRULENCE,
STRESS, …) regardless of `include_element_types`. This preserves the evidence
typing the downstream module requires. Fixed column order:

| Column | Meaning |
|--------|---------|
| `genome_id` | genome the hit belongs to |
| `element_symbol` | gene/mutation symbol (matrix column key for AMR) |
| `element_name` | human-readable element name |
| `scope` | AMRFinderPlus scope (`core` / `plus`) |
| `element_type` | `AMR` / `VIRULENCE` / `STRESS` / … |
| `element_subtype` | `AMR` (acquired) / `POINT` (mutation) / … |
| `feature_kind` | derived: `acquired_gene` or `point_mutation` |
| `class` | drug/stress class |
| `subclass` | drug/stress subclass |
| `method` | AMRFinderPlus Method (`EXACTX`, `BLASTX`, `POINTX`, `PARTIALX`, `HMM`, …) |
| `pct_identity` | % identity to reference (float) |
| `pct_coverage` | % coverage of reference (float) |
| `contig_id` | contig the hit is on |
| `start`, `stop`, `strand` | coordinates on the contig |
| `protein_id` | protein identifier (if any) |
| `target_length`, `reference_length`, `alignment_length` | lengths |
| `closest_reference_accession`, `closest_reference_name` | closest reference |
| `hmm_accession`, `hmm_description` | HMM info (if method HMM) |

Rows are sorted by `(genome_id, element_type, element_symbol, contig_id, start,
method)`.

---

## 5. `feature_schema.json` — versioned column manifest

```jsonc
{
  "schema_version": "1.0.0",
  "pipeline_version": "0.1.0",
  "generated_at": "2026-07-18T00:00:00Z",
  "matrix": {
    "id_column": "genome_id",
    "value_encoding": "presence_absence",
    "value_domain": [0, 1],
    "dtype": "int8",
    "n_genomes": 3,
    "n_features": 3,
    "included_element_types": ["AMR"],
    "column_order": "ascending lexicographic by element_symbol",
    "unseen_feature_policy": "dropped_and_logged",
    "unseen_feature_policy_detail": "..."
  },
  "provenance": {
    "organism": "Escherichia",
    "use_plus": true,
    "amrfinderplus_software_version": "4.0.x",
    "amrfinderplus_database_version": "2026-..."
  },
  "columns": [
    {
      "column": "gyrA_S83L",
      "element_symbol": "gyrA_S83L",
      "feature_kind": "point_mutation",
      "element_type": "AMR",
      "element_types_observed": ["AMR"],
      "element_subtypes_observed": ["POINT"],
      "drug_classes": ["QUINOLONE"],
      "drug_subclasses": ["QUINOLONE"],
      "methods_observed": ["POINTX"],
      "n_genomes_present": 1
    }
  ]
}
```

The `columns` array is **1:1 and in the same order** as the matrix feature
columns — the schema is generated *from* the matrix column list, so they can
never disagree. `feature_kind` is `point_mutation` when any observed subtype is
`POINT`, else `acquired_gene`.

---

## 6. Unseen genes at inference time

The matrix column set is **fixed to `feature_schema.json`**. When a new genome is
annotated against this schema:

- A known column (in the schema) **absent** in the new genome → encoded `0`.
- A known column **present** → encoded `1`.
- A gene/mutation **not in the schema** (novel to this corpus) → **dropped from
  the feature vector and logged**; it does **not** add a column. The matrix
  width never grows from unseen genomes, so a trained downstream model keeps a
  stable input dimensionality.

This policy is recorded as `matrix.unseen_feature_policy` (default
`dropped_and_logged`). Building the inference-time mapping/logging is a Module
2/3 concern; Module 1 defines and documents the contract and emits the fixed
schema.

---

## 7. `run_manifest.json` — reproducibility record

Contains: `run_id`, pipeline/schema versions, `started_at`/`finished_at`, the
fully-resolved `config`, a `tools` block (AMRFinderPlus software + database
version and path, BLAST+/HMMER versions, Python/pandas/pyarrow, platform),
`counts` (discovered / in-matrix / features / hits / status breakdown), and a
per-genome `inputs` list with `sha256`, `bytes`, `seq_type`, `status`,
`included_in_matrix`, `n_hits`, and any `error`.

The manifest is **provenance, not a model input** — its timestamps and
per-genome `annotation_seconds` legitimately vary between otherwise-identical
runs. The feature matrix and schema do not.

---

## 8. Determinism guarantees

Given identical inputs, config, and AMRFinderPlus software + database versions:

- `features_binary.csv` is **byte-identical** across runs.
- `features_binary.parquet` and `features_long.parquet` are byte-identical.
- `feature_schema.json` is identical except for the `generated_at` timestamp.
- `run_id` is a deterministic hash of the input checksums plus the parameters
  that affect output (organism, `--plus`, included types, tool/db versions).

Guaranteed by: sorted file discovery, sorted feature columns, fixed long-table
sort key, fixed schema field order, and a version-aware annotation cache that
re-runs a genome whenever any output-affecting input changes.

---

## 9. AMRFinderPlus column mapping (version tolerance)

AMRFinderPlus renamed several output columns between 3.x and 4.x. The parser
accepts **both**; canonical field ← accepted headers (4.x first):

| Canonical field | 4.x header | 3.x header |
|-----------------|-----------|-----------|
| `element_symbol` | `Element symbol` | `Gene symbol` |
| `element_name` | `Element name` | `Sequence name` |
| `element_type` | `Type` | `Element type` |
| `element_subtype` | `Subtype` | `Element subtype` |
| `pct_coverage` | `% Coverage of reference` | `% Coverage of reference sequence` |
| `pct_identity` | `% Identity to reference` | `% Identity to reference sequence` |
| `closest_reference_accession` | `Closest reference accession` | `Accession of closest sequence` |
| `closest_reference_name` | `Closest reference name` | `Name of closest sequence` |
| `hmm_accession` | `HMM accession` | `HMM id` |
| `protein_id` | `Protein id` | `Protein identifier` |

`Class`, `Subclass`, `Method`, `Scope`, `Contig id`, `Start`, `Stop`, `Strand`,
`HMM description` are unchanged. Required columns (`element_symbol`,
`element_type`, `element_subtype`, `method`) missing after alias resolution is a
hard parse error.
