"""Stage 4 - Feature matrix and long-format provenance table.

Two outputs are built here:

* **Binary matrix** (primary): rows = genomes, columns = the sorted union of
  element symbols observed across the corpus (restricted to the configured
  element types), values = 0/1 presence. Every genome in ``matrix_genome_ids``
  gets a row, including genomes with zero hits (all-zero row).

* **Long table** (provenance): one row per AMRFinderPlus hit, retaining the full
  per-hit metadata (element type, subtype, method, class, identity, coverage,
  coordinates, ...). This is intentionally *not* filtered by element type -- the
  downstream evidence-typing module needs the complete record.

Determinism: genomes are ordered by the caller (sorted), feature columns are
sorted ascending by symbol, and the long table is sorted by a fixed key.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import constants


def collect_feature_symbols(
    records: list[dict[str, Any]],
    include_element_types: tuple[str, ...],
) -> list[str]:
    """Sorted union of element symbols across records within the given types."""
    include = {t.upper() for t in include_element_types}
    symbols: set[str] = set()
    for rec in records:
        etype = (rec.get(constants.FIELD_ELEMENT_TYPE) or "").upper()
        symbol = rec.get(constants.FIELD_ELEMENT_SYMBOL)
        if symbol and etype in include:
            symbols.add(symbol)
    return sorted(symbols)


def build_binary_matrix(
    records: list[dict[str, Any]],
    matrix_genome_ids: list[str],
    include_element_types: tuple[str, ...],
) -> pd.DataFrame:
    """Build the genome x symbol presence/absence matrix.

    ``matrix_genome_ids`` defines the exact row set and order; any genome with
    no qualifying hit becomes an all-zero row. Column order is sorted symbols.
    """
    feature_symbols = collect_feature_symbols(records, include_element_types)
    include = {t.upper() for t in include_element_types}

    # genome_id -> set of present symbols (restricted to included types).
    present: dict[str, set[str]] = {gid: set() for gid in matrix_genome_ids}
    for rec in records:
        gid = rec.get("genome_id")
        if gid not in present:
            continue
        etype = (rec.get(constants.FIELD_ELEMENT_TYPE) or "").upper()
        symbol = rec.get(constants.FIELD_ELEMENT_SYMBOL)
        if symbol and etype in include:
            present[gid].add(symbol)

    data = {constants.ID_COLUMN: list(matrix_genome_ids)}
    for symbol in feature_symbols:
        data[symbol] = [
            1 if symbol in present[gid] else 0 for gid in matrix_genome_ids
        ]

    df = pd.DataFrame(data)
    # Fix dtypes: id as string, features as int8 (compact, stable).
    df[constants.ID_COLUMN] = df[constants.ID_COLUMN].astype("string")
    for symbol in feature_symbols:
        df[symbol] = df[symbol].astype("int8")
    return df


def build_long_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the per-hit provenance table with a fixed column order.

    Missing canonical fields are filled with NA so the schema is stable
    regardless of which optional AMRFinderPlus columns a given run produced.
    """
    columns = list(constants.LONG_TABLE_COLUMNS)
    if not records:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})

    rows = []
    for rec in records:
        rows.append({c: rec.get(c) for c in columns})
    df = pd.DataFrame(rows, columns=columns)

    # Deterministic ordering. Sort by a stable, human-meaningful key; NA-safe.
    sort_cols = [
        "genome_id",
        constants.FIELD_ELEMENT_TYPE,
        constants.FIELD_ELEMENT_SYMBOL,
        constants.FIELD_CONTIG_ID,
        constants.FIELD_START,
        constants.FIELD_METHOD,
    ]
    df = df.sort_values(
        by=sort_cols, kind="stable", na_position="last"
    ).reset_index(drop=True)
    return df
