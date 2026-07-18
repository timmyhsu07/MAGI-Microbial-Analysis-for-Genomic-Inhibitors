#!/usr/bin/env bash
#
# setup_amrfinder.sh - install AMRFinderPlus, download its database, and RECORD
# the exact versions for reproducibility.
#
# Commands here were verified against the official NCBI AMRFinderPlus wiki
# (Install-with-bioconda / Upgrading) in July 2026:
#   install : conda create ... -c bioconda ncbi-amrfinderplus
#   database: amrfinder -u   (default location) OR amrfinder_update -d <dir>
#
# This script does NOT pin a hardcoded tool version -- it installs the current
# bioconda release and writes whatever versions actually got installed into a
# manifest, which is the reproducibility contract the pipeline relies on.
#
# Usage:
#   scripts/setup_amrfinder.sh                 # env 'amrfinder', default DB dir
#   AMRFINDER_ENV=amr AMRFINDER_DB_DIR=./amrdb scripts/setup_amrfinder.sh
#
# Env vars:
#   AMRFINDER_ENV     conda/mamba env name           (default: amrfinder)
#   AMRFINDER_DB_DIR  custom database directory       (default: bundled/default)
#   SETUP_MANIFEST    where to write versions         (default: ./setup_versions.txt)
set -euo pipefail

ENV_NAME="${AMRFINDER_ENV:-amrfinder}"
DB_DIR="${AMRFINDER_DB_DIR:-}"
MANIFEST="${SETUP_MANIFEST:-setup_versions.txt}"

# --- pick a conda-compatible front-end --------------------------------------
if command -v micromamba >/dev/null 2>&1; then
  MGR=micromamba
elif command -v mamba >/dev/null 2>&1; then
  MGR=mamba
elif command -v conda >/dev/null 2>&1; then
  MGR=conda
else
  echo "ERROR: need micromamba, mamba, or conda on PATH. Install Miniforge:" >&2
  echo "  https://github.com/conda-forge/miniforge" >&2
  exit 1
fi
echo "Using package manager: $MGR"

run_in_env() {  # run a command inside the created env, manager-agnostically
  case "$MGR" in
    micromamba) micromamba run -n "$ENV_NAME" "$@" ;;
    *)          "$MGR" run -n "$ENV_NAME" "$@" ;;
  esac
}

# --- 1. install AMRFinderPlus ------------------------------------------------
echo ">> Creating env '$ENV_NAME' with ncbi-amrfinderplus (bioconda)..."
"$MGR" create -y -c conda-forge -c bioconda --strict-channel-priority \
  -n "$ENV_NAME" ncbi-amrfinderplus

# --- 2. download / update the database --------------------------------------
if [ -n "$DB_DIR" ]; then
  echo ">> Downloading AMRFinderPlus database into: $DB_DIR"
  mkdir -p "$DB_DIR"
  run_in_env amrfinder_update -d "$DB_DIR"
  DB_LATEST="$DB_DIR/latest"
else
  echo ">> Downloading AMRFinderPlus database into the install's default location"
  run_in_env amrfinder -u
  DB_LATEST=""  # tool uses its bundled/default DB path
fi

# --- 3. record versions ------------------------------------------------------
echo ">> Recording versions to: $MANIFEST"
{
  echo "# AMRFinderPlus setup manifest"
  echo "generated_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "env_name: $ENV_NAME"
  echo "package_manager: $MGR"
  echo "database_dir: ${DB_DIR:-<default>}"
  echo -n "amrfinder_software_version: "; run_in_env amrfinder --version 2>/dev/null || echo unknown
  if [ -n "$DB_LATEST" ] && [ -f "$DB_LATEST/version.txt" ]; then
    echo -n "amrfinder_database_version: "; cat "$DB_LATEST/version.txt"
  else
    echo -n "amrfinder_database_version: "; run_in_env amrfinder --database_version 2>/dev/null || echo unknown
  fi
  echo -n "blastn_version: "; run_in_env blastn -version 2>/dev/null | head -1 || echo unknown
  echo -n "blastx_version: "; run_in_env blastx -version 2>/dev/null | head -1 || echo unknown
  echo -n "hmmer_version: "; run_in_env hmmsearch -h 2>/dev/null | grep -m1 '# HMMER' || echo unknown
} | tee "$MANIFEST"

echo
echo "Done. To run the pipeline against this install, either activate the env:"
echo "    conda activate $ENV_NAME    # (or: micromamba activate $ENV_NAME)"
if [ -n "$DB_LATEST" ]; then
  echo "and pass --database-dir '$DB_LATEST' to genome_reader (or set database_dir in config)."
fi
