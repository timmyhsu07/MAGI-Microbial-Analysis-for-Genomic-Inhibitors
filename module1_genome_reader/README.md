# Module 1: Genome Reader

Turns a directory of assembled bacterial genome FASTA files into a reproducible
machine-learning feature matrix of antimicrobial-resistance (AMR)
determinants, using [NCBI AMRFinderPlus](https://github.com/ncbi/amr) as the
annotation engine.

This module only detects and tabulates resistance determinants already present
in an assembled genome. Modeling, calibration, decision rules, and the UI live
in Modules 2 and 3.

---

## What it produces

From `input_dir/` (one genome FASTA per file) to `output_dir/`:

- `features_binary.parquet` / `.csv` — genome × feature **presence/absence** (0/1) matrix (primary output)
- `features_long.parquet` — per-hit rich table (full AMRFinderPlus metadata; provenance)
- `feature_schema.json` — versioned column manifest
- `run_manifest.json` — tool/db/dependency versions, params, input checksums, per-genome status

See [OUTPUT_FORMAT_SPEC.md](OUTPUT_FORMAT_SPEC.md) for the full contract.

---

## Layout

```
module1_genome_reader/
├─ src/genome_reader/       # the pipeline package
│  ├─ constants.py          # schema version, column aliases (4.x/3.x), organism list
│  ├─ config.py             # Config dataclass, YAML/JSON load, validation
│  ├─ fasta.py              # FASTA read/validate, nucleotide-vs-protein detection
│  ├─ discovery.py          # stage 1: scan dir, extract ids, validate
│  ├─ versions.py           # capture tool/db/dep versions
│  ├─ annotate.py           # stage 2: run AMRFinderPlus (parallel, cached, injectable)
│  ├─ parse.py              # stage 3: version-tolerant TSV parsing
│  ├─ matrix.py             # stage 4: binary matrix + long table
│  ├─ schema.py             # stage 5: feature_schema.json
│  ├─ manifest.py           # run_manifest.json
│  ├─ pipeline.py           # end-to-end orchestration
│  └─ cli.py / __main__.py  # `python -m genome_reader`
├─ scripts/
│  ├─ setup_amrfinder.sh    # install AMRFinderPlus + DB, record versions
│  ├─ fetch_test_fixtures.sh# (optional) download tiny real genomes
│  └─ demo_mock_run.py      # run the whole pipeline with mock TSVs (no tool needed)
├─ config/config.example.yaml
├─ tests/                   # unit + end-to-end tests (mocked AMRFinderPlus)
├─ environment.yml          # conda env (AMRFinderPlus + Python stack)
├─ pyproject.toml
├─ OUTPUT_FORMAT_SPEC.md
└─ README.md
```

---

## Install

### 1. AMRFinderPlus + database (required for real runs)

The commands below are verified against the official NCBI wiki
([Install-with-bioconda](https://github.com/ncbi/amr/wiki/Install-with-bioconda),
[Upgrading](https://github.com/ncbi/amr/wiki/Upgrading)). The setup script wraps
them and **records the resolved versions**:

```bash
# Installs ncbi-amrfinderplus into a conda/mamba/micromamba env, downloads the
# database, and writes setup_versions.txt.
scripts/setup_amrfinder.sh
#   AMRFINDER_ENV=amrfinder            (env name)
#   AMRFINDER_DB_DIR=./amrdb           (optional custom DB dir)
```

Equivalently, by hand:

```bash
conda create -y -c conda-forge -c bioconda --strict-channel-priority \
  -n amrfinder ncbi-amrfinderplus
conda activate amrfinder
amrfinder -u                 # download the database (one-time)
```

Or create the full environment (AMRFinderPlus + Python deps) at once:

```bash
conda env create -f environment.yml
conda activate genome-reader
amrfinder -u
```

### 2. The pipeline package

Inside an env that has `pandas`, `pyarrow`, `pyyaml`:

```bash
pip install -e .            # exposes the `genome-reader` command
# or run without installing:
PYTHONPATH=src python -m genome_reader --help
```

---

## Quickstart

```bash
genome-reader \
  --input-dir data/genomes \
  --output-dir out \
  --organism Escherichia \
  --workers 8
```

or with a config file (CLI flags override file values):

```bash
cp config/config.example.yaml config.yaml   # edit paths/organism
genome-reader --config config.yaml
```

`--organism` enables organism-specific **point-mutation** screening and must be
a recognized AMRFinderPlus value — list them with:

```bash
genome-reader --list-organisms
```

### Try it now, without AMRFinderPlus

The repo ships mock AMRFinderPlus TSVs so you can see the full output shape
instantly:

```bash
python scripts/demo_mock_run.py         # writes example_run/out/*
cat example_run/out/features_binary.csv
```

---

## Configuration reference

| Key | Default | Meaning |
|-----|---------|---------|
| `input_dir` | — (required) | directory of genome FASTA files |
| `output_dir` | — (required) | where artifacts are written |
| `organism` | `null` | AMRFinderPlus `--organism`; enables point mutations |
| `amrfinder_bin` | `amrfinder` | path to the executable |
| `database_dir` | `null` | AMRFinderPlus DB dir (`…/latest`); null = bundled |
| `use_plus` | `true` | add `--plus` (stress/virulence screening) |
| `amrfinder_threads` | `1` | threads per AMRFinderPlus run |
| `workers` | `4` | genomes annotated concurrently |
| `reuse_cache` | `true` | skip genomes whose cached result is still valid |
| `cache_dir` | `null` | default `<output_dir>/cache` |
| `allow_partial` | `false` | if false, any genome failure aborts the run loudly |
| `protein_handling` | `skip` | `skip` or `annotate` detected protein FASTAs |
| `include_element_types` | `["AMR"]` | element types that become matrix columns |
| `unseen_feature_policy` | `dropped_and_logged` | recorded contract for novel genes |

Every key has a matching CLI flag (`--input-dir`, `--organism`, `--no-plus`,
`--no-cache`, `--allow-partial`, `--include-element-types`, …).

---

## Reproducibility

- **Pinned & recorded:** `run_manifest.json` captures AMRFinderPlus software +
  database versions, BLAST+/HMMER, Python/pandas/pyarrow, the resolved config,
  and per-genome input SHA-256 checksums.
- **Deterministic:** sorted discovery, sorted feature columns, fixed schema
  order → same inputs produce a **byte-identical** feature matrix. `run_id` is a
  deterministic hash of inputs + output-affecting parameters.
- **Caching:** per-genome results are cached and keyed on the input checksum,
  organism, `--plus`, mode, and AMRFinderPlus software + database version — so a
  rerun skips completed genomes but automatically re-runs any genome affected by
  a tool/DB upgrade.
- **Fails loudly:** malformed/empty FASTA, duplicate genome IDs, missing
  database, unknown organism, and (by default) any genome annotation failure all
  abort with a clear message.

---

## Testing

Tests use mocked AMRFinderPlus TSVs (in `tests/data/mock_amrfinder/`) and need
no network or AMRFinderPlus install:

```bash
pip install pytest
PYTHONPATH=src pytest tests -v
```

The end-to-end tests assert matrix **shape**, **determinism** (run twice →
byte-identical), **all-zero-row** handling for hit-free genomes, protein-skip
and failure behavior, and **schema validity**.

For an optional smoke test against the *real* tool, download tiny public
genomes with `scripts/fetch_test_fixtures.sh` and run the CLI against them.

---

## Notes / assumptions

- Nucleotide assemblies are the primary input (`-n`). Files detected as protein
  are skipped by default (`protein_handling: skip`) or can be run with `-p`
  (`protein_handling: annotate`) — note point mutations require nucleotide input.
- `.gz`-compressed FASTAs are supported (decompressed transparently for
  AMRFinderPlus).
- AMRFinderPlus 4.x and 3.x TSV column spellings are both accepted; see the
  mapping table in [OUTPUT_FORMAT_SPEC.md](OUTPUT_FORMAT_SPEC.md#9-amrfinderplus-column-mapping-version-tolerance).
