#!/usr/bin/env python3
"""Fetch a small, real, labelled E. coli cohort from BV-BRC for the demo.

This is the Phase 1 bottleneck-buster: it turns BV-BRC's public data API into a
Module 1-ready input directory plus Module 2-ready label tables, with zero
third-party dependencies (stdlib urllib/csv/json only, so it can't break on a
fresh machine minutes before a demo).

What it produces under --out-dir (default: data/bvbrc_ecoli/):

  genomes/<genome_id>.fna   assembled contigs per isolate  -> Module 1 input_dir
  labels.csv                genome_id + R/S/I per drug      -> Module 2 labels
  target_genes.csv          genome_id + gyrA,parC,ftsI,rpsL -> Module 2 gate input
  cohort_manifest.json      full provenance (queries, counts, per-genome QC)

Data model (verified against the live API):
  * Labels come from the `genome_amr` collection, filtered to E. coli
    (taxon_id=562) and the three modelled drugs. One row is one lab test, so a
    genome can carry several rows per drug; we resolve them by majority vote and
    drop genuine ties.
  * The assembled FASTA comes from the `genome_sequence` collection with
    `http_accept=application/dna+fasta`. `genome_id` (e.g. 562.56783) is the
    join key across all three artifacts and matches Module 1's genome_id (it
    strips the .fna extension back to the same string).

Honesty note on target_genes.csv: gyrA/parC/ftsI/rpsL are the drugs' molecular
*targets* (essential/near-universal genes in a viable E. coli), NOT resistance
determinants. AMRFinderPlus only reports a target gene when it carries a
resistance mutation, so target *presence* can't be read off the AMR matrix. For
a QC'd E. coli assembly we therefore record them as present (1) with this
documented assumption; the deterministic gate consequently does not fire on this
real cohort (that branch stays demonstrated in Module 3's mock pipeline). A real
deployment would confirm target presence with a dedicated housekeeping-gene
annotation.

Usage:
  python scripts/fetch_bvbrc_ecoli.py                 # plan + download 40 isolates
  python scripts/fetch_bvbrc_ecoli.py --max-genomes 60
  python scripts/fetch_bvbrc_ecoli.py --plan-only     # build label plan, no FASTA
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

API = "https://www.bv-brc.org/api"
ECOLI_TAXON = 562
DRUGS = ["ampicillin", "ciprofloxacin", "gentamicin"]  # Module 2's covered set
# Drug -> molecular target gene(s) the deterministic gate checks (see
# module2_predictor/contracts/config.yaml). The union becomes target_genes.csv's
# columns; every gene named in any drug's config target_genes list must appear.
TARGET_GENES = ["gyrA", "parC", "ftsI", "rpsL"]

_PHENO_TO_LABEL = {"Resistant": "R", "Susceptible": "S", "Intermediate": "I"}
# E. coli genome length sanity window (bp). ~4.6 Mbp typical; drafts vary, so be
# lenient but reject obvious junk / contamination / wrong-organism assemblies.
MIN_GENOME_BP = 3_500_000
MAX_GENOME_BP = 7_500_000


def _get(path: str, accept: str = "application/json", retries: int = 4, timeout: int = 90) -> bytes:
    """GET an API path (RQL already in `path`), with retry/backoff."""
    url = f"{API}/{path}"
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url,
            headers={"Accept": accept, "User-Agent": "genome-firewall-demo/1.0 (+hacknation)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last = exc
            wait = 2 ** attempt
            print(f"    ! request failed ({exc}); retry {attempt + 1}/{retries} in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


def fetch_labels(limit_per_drug: int) -> dict[str, dict[str, str]]:
    """Return {genome_id: {drug: 'R'|'S'|'I'}} resolved by majority vote.

    Pulls Resistant/Susceptible/Intermediate genome_amr rows for E. coli per
    drug, then collapses the many-rows-per-(genome,drug) reality to one label.
    """
    raw: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for drug in DRUGS:
        rql = (
            f"genome_amr/?and(eq(taxon_id,{ECOLI_TAXON}),eq(antibiotic,{drug}))"
            f"&select(genome_id,resistant_phenotype)&limit({limit_per_drug})"
        )
        data = _get(rql, accept="text/tsv").decode("utf-8", "replace")
        rows = list(csv.DictReader(io.StringIO(data), delimiter="\t"))
        kept = 0
        for row in rows:
            gid = (row.get("genome_id") or "").strip()
            label = _PHENO_TO_LABEL.get((row.get("resistant_phenotype") or "").strip())
            if gid and label:
                raw[gid][drug].append(label)
                kept += 1
        print(f"  {drug:14s} {len(rows):6d} rows -> {kept:6d} usable", file=sys.stderr)

    resolved: dict[str, dict[str, str]] = {}
    for gid, per_drug in raw.items():
        out: dict[str, str] = {}
        for drug, labels in per_drug.items():
            counts = Counter(labels)
            top, n = counts.most_common(1)[0]
            # Majority wins; drop the label on a genuine tie between R and S.
            if list(counts.values()).count(n) == 1:
                out[drug] = top
        if out:
            resolved[gid] = out
    return resolved


def _score(labels: dict[str, str]) -> tuple[int, int]:
    """Sort key: prefer genomes labelled for all 3 drugs, then more R/S calls."""
    rs = sum(1 for v in labels.values() if v in ("R", "S"))
    return (len(labels), rs)


def select_cohort(resolved: dict[str, dict[str, str]], n: int) -> list[str]:
    """Greedily pick ~n genomes that (a) are richly labelled and (b) give each
    drug both R and S examples so no drug gets skipped for one-class data."""
    ranked = sorted(resolved, key=lambda g: _score(resolved[g]), reverse=True)
    chosen: list[str] = []
    seen_class: dict[tuple[str, str], int] = Counter()

    # Pass 1: guarantee minority-class coverage per drug (2 of each R/S).
    for want in ("R", "S"):
        for drug in DRUGS:
            for g in ranked:
                if len(chosen) >= n:
                    break
                if g in chosen:
                    continue
                if resolved[g].get(drug) == want and seen_class[(drug, want)] < 2:
                    chosen.append(g)
                    for d, v in resolved[g].items():
                        seen_class[(d, v)] += 1
    # Pass 2: fill remaining slots with the richest-labelled genomes.
    for g in ranked:
        if len(chosen) >= n:
            break
        if g not in chosen:
            chosen.append(g)
    return chosen[:n]


def _fasta_stats(text: str) -> tuple[int, int]:
    """(total_bp, n_contigs) for a FASTA string."""
    contigs = 0
    total = 0
    for line in text.splitlines():
        if line.startswith(">"):
            contigs += 1
        else:
            total += len(line.strip())
    return total, contigs


def download_fasta(genome_id: str, dest: Path) -> tuple[int, int]:
    """Download one genome's assembled contigs; return (total_bp, n_contigs)."""
    rql = f"genome_sequence/?eq(genome_id,{genome_id})&limit(5000)&http_accept=application/dna+fasta"
    text = _get(rql, accept="application/dna+fasta").decode("utf-8", "replace")
    total, contigs = _fasta_stats(text)
    if not text.lstrip().startswith(">"):
        raise RuntimeError(f"{genome_id}: response is not FASTA")
    dest.write_text(text, encoding="utf-8")
    return total, contigs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/bvbrc_ecoli", type=Path)
    ap.add_argument("--max-genomes", type=int, default=40)
    ap.add_argument("--limit-per-drug", type=int, default=20000,
                    help="max genome_amr rows to pull per drug when building the label plan")
    ap.add_argument("--plan-only", action="store_true", help="build labels only; skip FASTA download")
    args = ap.parse_args()

    out = args.out_dir
    genomes_dir = out / "genomes"
    genomes_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Fetching E. coli AMR phenotypes from BV-BRC ...", file=sys.stderr)
    resolved = fetch_labels(args.limit_per_drug)
    all3 = sum(1 for v in resolved.values() if len(v) == 3)
    print(f"      {len(resolved)} genomes with >=1 resolved label ({all3} with all 3 drugs)", file=sys.stderr)

    print(f"[2/4] Selecting a cohort of up to {args.max_genomes} ...", file=sys.stderr)
    cohort = select_cohort(resolved, args.max_genomes)
    print(f"      selected {len(cohort)} candidate genomes", file=sys.stderr)

    manifest: dict = {
        "source": "BV-BRC data API (genome_amr + genome_sequence)",
        "api": API,
        "species": "Escherichia coli",
        "taxon_id": ECOLI_TAXON,
        "drugs": DRUGS,
        "target_genes": TARGET_GENES,
        "target_genes_assumption": "present(1) for QC'd E. coli; targets are essential genes, "
                                   "not readable from the AMR matrix (see script docstring)",
        "label_resolution": "majority vote over per-test genome_amr rows; ties dropped",
        "qc_window_bp": [MIN_GENOME_BP, MAX_GENOME_BP],
        "genomes": [],
    }

    accepted: list[str] = []
    if args.plan_only:
        accepted = cohort
        for gid in cohort:
            manifest["genomes"].append({"genome_id": gid, "labels": resolved[gid], "downloaded": False})
    else:
        print("[3/4] Downloading assembled FASTAs + QC ...", file=sys.stderr)
        for gid in cohort:
            dest = genomes_dir / f"{gid}.fna"
            try:
                total, contigs = download_fasta(gid, dest)
            except Exception as exc:  # noqa: BLE001 - keep going, report per-genome
                print(f"      x {gid}: download failed ({exc})", file=sys.stderr)
                dest.unlink(missing_ok=True)
                manifest["genomes"].append({"genome_id": gid, "labels": resolved[gid],
                                            "downloaded": False, "error": str(exc)})
                continue
            ok = MIN_GENOME_BP <= total <= MAX_GENOME_BP
            status = "ok" if ok else "qc_fail"
            print(f"      {'+' if ok else '-'} {gid}: {total:,} bp / {contigs} contigs [{status}]", file=sys.stderr)
            if not ok:
                dest.unlink(missing_ok=True)
            else:
                accepted.append(gid)
            manifest["genomes"].append({"genome_id": gid, "labels": resolved[gid],
                                        "downloaded": ok, "total_bp": total, "n_contigs": contigs})

    # --- Write Module 2 label tables for the accepted genomes ---------------
    print(f"[4/4] Writing labels.csv / target_genes.csv for {len(accepted)} genomes ...", file=sys.stderr)
    with (out / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["genome_id", *DRUGS])
        for gid in accepted:
            w.writerow([gid, *(resolved[gid].get(d, "") for d in DRUGS)])
    with (out / "target_genes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["genome_id", *TARGET_GENES])
        for gid in accepted:
            w.writerow([gid, *([1] * len(TARGET_GENES))])

    # Per-drug class balance over the accepted cohort (what Module 2 will train on).
    balance = {d: dict(Counter(resolved[g].get(d, "") for g in accepted)) for d in DRUGS}
    manifest["n_accepted"] = len(accepted)
    manifest["class_balance"] = balance
    (out / "cohort_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDONE.", file=sys.stderr)
    print(f"  genomes: {genomes_dir}  ({len(accepted)} FASTAs)", file=sys.stderr)
    print(f"  labels:  {out / 'labels.csv'}", file=sys.stderr)
    print(f"  targets: {out / 'target_genes.csv'}", file=sys.stderr)
    print(f"  class balance (R/S/I per drug): {json.dumps(balance)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
