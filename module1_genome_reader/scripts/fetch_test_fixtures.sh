#!/usr/bin/env bash
#
# fetch_test_fixtures.sh - download a tiny set of small public bacterial genomes
# for OPTIONAL end-to-end integration against the REAL AMRFinderPlus. The
# committed unit/E2E tests do NOT need this (they use mocked AMRFinderPlus TSVs
# under tests/data/mock_amrfinder). Use this only to smoke-test a real install.
#
# Requires the NCBI 'datasets' CLI (bioconda: `conda install -c conda-forge -c
# bioconda ncbi-datasets-cli`) OR curl access to NCBI.
#
# Usage: scripts/fetch_test_fixtures.sh [dest_dir]
set -euo pipefail

DEST="${1:-fixtures/genomes}"
mkdir -p "$DEST"

# Two small, complete RefSeq assemblies (E. coli reference + a small genome).
# Kept short and explicit so a reviewer can see exactly what is downloaded.
ASSEMBLIES=(
  "GCF_000005845.2"   # Escherichia coli str. K-12 substr. MG1655
  "GCF_000006945.2"   # Salmonella enterica Typhimurium LT2
)

if command -v datasets >/dev/null 2>&1; then
  for acc in "${ASSEMBLIES[@]}"; do
    echo ">> datasets download genome accession $acc"
    tmpzip="$(mktemp -t amr_XXXX).zip"
    datasets download genome accession "$acc" --include genome --filename "$tmpzip"
    unzip -o -q "$tmpzip" -d "$DEST/_unzip_$acc"
    # Flatten: copy the assembly .fna to <acc>.fna
    found="$(find "$DEST/_unzip_$acc" -name '*.fna' | head -1)"
    cp "$found" "$DEST/$acc.fna"
    rm -rf "$DEST/_unzip_$acc" "$tmpzip"
    echo "   -> $DEST/$acc.fna"
  done
else
  echo "ERROR: the NCBI 'datasets' CLI was not found on PATH." >&2
  echo "Install it, or download the assemblies manually and place their .fna" >&2
  echo "files in '$DEST' named <accession>.fna. Accessions:" >&2
  printf '  %s\n' "${ASSEMBLIES[@]}" >&2
  exit 1
fi

echo
echo "Fetched $(ls -1 "$DEST"/*.fna 2>/dev/null | wc -l | tr -d ' ') genome(s) into $DEST"
echo "Run: genome_reader --input-dir '$DEST' --output-dir out --organism Escherichia"
