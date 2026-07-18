#!/usr/bin/env python3
"""Generate the Module 2 fixture corpus.

Everything downstream of Module 1 needs a labelled feature matrix to train and
test against, and we can't ship the real AMRFinderPlus database or a real
phenotype panel. So this builds a small, deterministic, *synthetic* corpus that
mimics the shape of Module 1's output: a binary AMR feature matrix plus a
matching feature_schema.json, a phenotype label table, and a molecular
target-gene table for the gate.

The corpus is intentionally structured so the rest of the module has something
real to chew on:

* Genomes come in clonal groups (a base strain re-sequenced a few times with a
  couple of feature flips), so genetic clustering has near-duplicates to
  collapse and the leakage test has multi-member groups to keep intact.
* Resistance is driven by a handful of genes with a little label noise, so the
  logistic models learn a real but imperfect signal.
* A few genomes are missing a drug's target gene, so the deterministic gate
  actually fires somewhere.

Re-run any time; output is a pure function of SEED.

    python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SEED = 20260718
HERE = pathlib.Path(__file__).resolve().parents[1]
OUT = HERE / "tests" / "fixtures"

# The AMR feature vocabulary (AMRFinderPlus element symbols). Point mutations
# carry the S83L / S80I style suffix; everything else is an acquired gene.
FEATURES = [
    "aac(3)-IIa",
    "aac(6')-Ib",
    "aadA1",
    "aph(3')-Ia",
    "blaCTX-M-15",
    "blaSHV-12",
    "blaTEM-1",
    "dfrA1",
    "gyrA_S83L",
    "parC_S80I",
    "qnrS1",
    "sul1",
    "sul2",
    "tet(A)",
]

# Drug classes per symbol — only used to populate the schema realistically.
DRUG_CLASS = {
    "aac(3)-IIa": "AMINOGLYCOSIDE",
    "aac(6')-Ib": "AMINOGLYCOSIDE",
    "aadA1": "AMINOGLYCOSIDE",
    "aph(3')-Ia": "AMINOGLYCOSIDE",
    "blaCTX-M-15": "BETA-LACTAM",
    "blaSHV-12": "BETA-LACTAM",
    "blaTEM-1": "BETA-LACTAM",
    "dfrA1": "TRIMETHOPRIM",
    "gyrA_S83L": "QUINOLONE",
    "parC_S80I": "QUINOLONE",
    "qnrS1": "QUINOLONE",
    "sul1": "SULFONAMIDE",
    "sul2": "SULFONAMIDE",
    "tet(A)": "TETRACYCLINE",
}

TARGET_GENES = ["gyrA", "parC", "ftsI", "rpsL"]

# Base strains: (name, carried AMR genes). Clonal replicates are derived from
# these with a couple of random feature flips.
STRAINS = [
    ("STR01", ["blaTEM-1", "sul1", "tet(A)"]),
    ("STR02", ["blaCTX-M-15", "gyrA_S83L", "parC_S80I", "aac(6')-Ib", "qnrS1"]),
    ("STR03", ["gyrA_S83L", "aac(3)-IIa", "aadA1"]),
    ("STR04", []),  # pan-susceptible
    ("STR05", ["blaSHV-12", "blaTEM-1", "sul2", "dfrA1"]),
    ("STR06", ["aac(3)-IIa", "aph(3')-Ia", "tet(A)"]),
    ("STR07", ["blaCTX-M-15", "gyrA_S83L", "parC_S80I", "aac(3)-IIa", "sul1", "tet(A)"]),
    ("STR08", ["blaTEM-1"]),
    ("STR09", ["gyrA_S83L", "parC_S80I"]),
]

REPLICATES = {
    "STR01": 6, "STR02": 5, "STR03": 5, "STR04": 6, "STR05": 4,
    "STR06": 5, "STR07": 4, "STR08": 4, "STR09": 3,
}

# Deterministic target-gene knockouts (which genome loses which target).
TARGET_KNOCKOUTS = {
    "STR04_ISO01": {"ftsI": 0},               # ampicillin target absent
    "STR04_ISO02": {"gyrA": 0, "parC": 0},    # both ciprofloxacin targets absent
    "STR08_ISO01": {"rpsL": 0},               # gentamicin target absent
}


def _phenotype(rng: np.random.Generator, present: set[str]) -> dict[str, str]:
    """Rule-driven phenotype with a little label noise (~8%)."""
    def call(is_resistant: bool) -> str:
        if rng.random() < 0.08:  # measurement / mechanism noise
            is_resistant = not is_resistant
        return "R" if is_resistant else "S"

    return {
        "ampicillin": call(bool(present & {"blaTEM-1", "blaCTX-M-15", "blaSHV-12"})),
        "ciprofloxacin": call("gyrA_S83L" in present),
        "gentamicin": call(bool(present & {"aac(3)-IIa"})),
    }


def build() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []          # feature matrix rows
    label_rows = []    # phenotype rows
    target_rows = []   # target-gene presence rows

    for name, genes in STRAINS:
        base = {f: (1 if f in genes else 0) for f in FEATURES}
        for i in range(REPLICATES[name]):
            gid = f"{name}_ISO{i + 1:02d}"
            vec = dict(base)
            # Clonal drift: flip a small number of features on replicates.
            if i > 0:
                for f in rng.choice(FEATURES, size=int(rng.integers(0, 3)), replace=False):
                    vec[f] = 1 - vec[f]
            present = {f for f, v in vec.items() if v}

            rows.append({"genome_id": gid, **vec})
            label_rows.append({"genome_id": gid, **_phenotype(rng, present)})

            # Targets are normally all present. A few genomes have a target
            # knocked out so the gate has a guaranteed firing for each drug:
            # ftsI absent -> ampicillin gate, rpsL absent -> gentamicin gate,
            # both gyrA+parC absent -> ciprofloxacin gate.
            targets = {g: 1 for g in TARGET_GENES}
            targets.update(TARGET_KNOCKOUTS.get(gid, {}))
            target_rows.append({"genome_id": gid, **targets})

    matrix = pd.DataFrame(rows).sort_values("genome_id").reset_index(drop=True)
    matrix["genome_id"] = matrix["genome_id"].astype("string")
    for f in FEATURES:
        matrix[f] = matrix[f].astype("int8")
    matrix = matrix[["genome_id", *FEATURES]]

    labels = pd.DataFrame(label_rows).sort_values("genome_id").reset_index(drop=True)
    targets_df = pd.DataFrame(target_rows).sort_values("genome_id").reset_index(drop=True)

    matrix.to_parquet(OUT / "features_binary.parquet", index=False)
    matrix.to_csv(OUT / "features_binary.csv", index=False)
    labels.to_csv(OUT / "labels.csv", index=False)
    targets_df.to_csv(OUT / "target_genes.csv", index=False)
    _write_schema(matrix)

    n_r = {d: int((labels[d] == "R").sum()) for d in ("ampicillin", "ciprofloxacin", "gentamicin")}
    print(f"wrote {len(matrix)} genomes, {len(FEATURES)} features -> {OUT}")
    print(f"resistant counts: {n_r}")


def _write_schema(matrix: pd.DataFrame) -> None:
    """Emit a feature_schema.json in Module 1's format for the fixture matrix."""
    columns = []
    for f in FEATURES:
        is_point = f.endswith(("S83L", "S80I"))
        columns.append(
            {
                "column": f,
                "element_symbol": f,
                "feature_kind": "point_mutation" if is_point else "acquired_gene",
                "element_type": "AMR",
                "element_types_observed": ["AMR"],
                "element_subtypes_observed": ["POINT"] if is_point else ["AMR"],
                "drug_classes": [DRUG_CLASS[f]],
                "drug_subclasses": [DRUG_CLASS[f]],
                "methods_observed": ["POINTX"] if is_point else ["EXACTX"],
                "n_genomes_present": int(matrix[f].sum()),
            }
        )
    schema = {
        "schema_version": "1.0.0",
        "pipeline_version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "matrix": {
            "id_column": "genome_id",
            "value_encoding": "presence_absence",
            "value_domain": [0, 1],
            "dtype": "int8",
            "n_genomes": int(len(matrix)),
            "n_features": len(FEATURES),
            "included_element_types": ["AMR"],
            "column_order": "ascending lexicographic by element_symbol",
            "unseen_feature_policy": "dropped_and_logged",
        },
        "provenance": {
            "organism": "Escherichia",
            "use_plus": True,
            "amrfinderplus_software_version": "4.0.19",
            "amrfinderplus_database_version": "2026-01-15.1",
            "synthetic_fixture": True,
        },
        "columns": columns,
    }
    (OUT / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
