"""Input readers and contract checks.

Module 2 uses Module 1's feature schema to define column order. The matrix may
contain additional columns, but training and inference are pinned to the schema
order so model coefficients remain auditable and stable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config

ID_COLUMN = "genome_id"
LABEL_VALUES = {"R", "S", "I", ""}


class InputError(ValueError):
    """Raised when an input artifact violates the published contract."""


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(obj: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    schema = read_json(cfg.inputs.feature_schema)
    feature_columns = feature_columns_from_schema(schema)
    features = load_feature_matrix(cfg.inputs.feature_matrix, feature_columns)
    labels = load_labels(cfg.inputs.labels, cfg)
    targets = load_targets(cfg.inputs.target_gene_table, cfg)
    return features, schema, labels, targets


def feature_columns_from_schema(schema: dict[str, Any]) -> list[str]:
    columns = schema.get("columns")
    if not isinstance(columns, list):
        raise InputError("feature_schema.json must contain a 'columns' list")
    names = [c.get("column") for c in columns if isinstance(c, dict)]
    if not names or not all(isinstance(c, str) for c in names):
        raise InputError("feature_schema columns must each name a string 'column'")
    return list(names)


def load_feature_matrix(path: str | Path, feature_columns: list[str]) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p, keep_default_na=False)
    else:
        raise InputError(f"unsupported feature matrix extension: {p.suffix}")

    if ID_COLUMN not in df.columns:
        raise InputError(f"feature matrix is missing required column '{ID_COLUMN}'")
    if df[ID_COLUMN].duplicated().any():
        dupes = df.loc[df[ID_COLUMN].duplicated(), ID_COLUMN].tolist()
        raise InputError(f"feature matrix genome_id values must be unique: {dupes[:5]}")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise InputError(f"feature matrix is missing schema column(s): {missing}")

    out = df[[ID_COLUMN, *feature_columns]].copy()
    out[ID_COLUMN] = out[ID_COLUMN].astype("string")
    for col in feature_columns:
        vals = set(pd.Series(out[col]).dropna().astype(int).unique().tolist())
        if vals - {0, 1}:
            raise InputError(f"feature matrix column '{col}' has values outside 0/1")
        out[col] = out[col].astype("int8")
    return out


def load_labels(path: str | Path, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    if ID_COLUMN not in df.columns:
        raise InputError(f"labels table is missing required column '{ID_COLUMN}'")
    if df[ID_COLUMN].duplicated().any():
        dupes = df.loc[df[ID_COLUMN].duplicated(), ID_COLUMN].tolist()
        raise InputError(f"labels genome_id values must be unique: {dupes[:5]}")

    label_drugs = [c for c in df.columns if c != ID_COLUMN]
    unknown = sorted(set(label_drugs) - set(cfg.drugs))
    if unknown:
        raise InputError(f"labels contain drug(s) not configured: {unknown}")
    missing = sorted(set(cfg.drugs) - set(label_drugs))
    if missing:
        raise InputError(f"labels are missing configured drug column(s): {missing}")

    for col in label_drugs:
        bad = sorted(set(df[col].astype(str).tolist()) - LABEL_VALUES)
        if bad:
            raise InputError(f"labels column '{col}' has invalid value(s): {bad}")
    return df


def load_targets(path: str | Path, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    if ID_COLUMN not in df.columns:
        raise InputError(f"target table is missing required column '{ID_COLUMN}'")
    if df[ID_COLUMN].duplicated().any():
        dupes = df.loc[df[ID_COLUMN].duplicated(), ID_COLUMN].tolist()
        raise InputError(f"target table genome_id values must be unique: {dupes[:5]}")

    required = sorted({g for d in cfg.drugs.values() for g in d["target_genes"]})
    missing = [g for g in required if g not in df.columns]
    if missing:
        raise InputError(f"target table is missing configured target gene(s): {missing}")

    for col in [c for c in df.columns if c != ID_COLUMN]:
        values = pd.Series(df[col]).dropna()
        coerced = pd.to_numeric(values, errors="coerce")
        if coerced.isna().any():
            raise InputError(f"target table column '{col}' must contain only 0/1")
        bad = set(coerced.astype(int).tolist()) - {0, 1}
        if bad:
            raise InputError(f"target table column '{col}' has values outside 0/1")
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int8")
    df[ID_COLUMN] = df[ID_COLUMN].astype("string")
    return df


def matrix_values(features: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return features[feature_columns].to_numpy(dtype=np.int8, copy=True)
