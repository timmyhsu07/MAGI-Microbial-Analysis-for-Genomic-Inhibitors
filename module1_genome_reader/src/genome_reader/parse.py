"""Stage 3 - Parse.

Turn an AMRFinderPlus TSV into a list of normalized hit records keyed by this
pipeline's canonical field names (see :mod:`constants`). The header-alias table
makes parsing tolerant of both 4.x and 3.x column spellings.

A genome with no hits yields an empty list (its TSV still has a header row);
that is expected and downstream code renders it as an all-zero matrix row.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from . import constants


class ParseError(ValueError):
    """Raised when a TSV cannot be parsed or is missing required columns."""


def _normalize(header: str) -> str:
    return header.strip().lower()


def resolve_header_map(headers: list[str]) -> dict[str, int]:
    """Map canonical field name -> column index for the given TSV headers.

    Only fields we recognize (present in ``HEADER_ALIASES``) are mapped; unknown
    columns are ignored. Raises :class:`ParseError` if any REQUIRED field is
    absent.
    """
    normalized = {_normalize(h): i for i, h in enumerate(headers)}
    mapping: dict[str, int] = {}
    for field_name, aliases in constants.HEADER_ALIASES.items():
        for alias in aliases:
            idx = normalized.get(_normalize(alias))
            if idx is not None:
                mapping[field_name] = idx
                break

    missing = [f for f in constants.REQUIRED_FIELDS if f not in mapping]
    if missing:
        raise ParseError(
            "AMRFinderPlus TSV is missing required column(s) "
            f"{missing}. Present headers: {headers}. This usually means an "
            "unexpected AMRFinderPlus output format; update HEADER_ALIASES in "
            "constants.py."
        )
    return mapping


def _to_number(value: str) -> float | None:
    value = value.strip()
    if value in ("", "NA", "N/A", "."):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _clean_str(value: str) -> str | None:
    value = value.strip()
    if value in ("", "NA", "N/A"):
        return None
    return value


def feature_kind_for(subtype: str | None) -> str:
    if subtype and subtype.strip().upper() == constants.SUBTYPE_POINT:
        return constants.FEATURE_KIND_POINT
    return constants.FEATURE_KIND_ACQUIRED


def parse_tsv(tsv_path: str | Path, genome_id: str) -> list[dict[str, Any]]:
    """Parse one AMRFinderPlus TSV into canonical hit records for ``genome_id``.

    Each record is a dict containing ``genome_id``, ``feature_kind``, and every
    canonical field present in the file (numeric fields as float-or-None, string
    fields as str-or-None).
    """
    path = Path(tsv_path)
    if not path.exists():
        raise ParseError(f"{path}: TSV does not exist")

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        try:
            headers = next(reader)
        except StopIteration:
            raise ParseError(f"{path}: file is empty (no header row)") from None

        mapping = resolve_header_map(headers)
        n_cols = len(headers)
        records: list[dict[str, Any]] = []

        for line_no, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == "" for cell in row):
                continue  # skip blank lines
            if len(row) < n_cols:
                # Ragged row: pad so index lookups are safe, but flag it loudly.
                raise ParseError(
                    f"{path}:{line_no}: row has {len(row)} fields, expected "
                    f"{n_cols}. File may be truncated or corrupted."
                )

            record: dict[str, Any] = {
                "genome_id": genome_id,
                "source_line": line_no,
            }
            for field_name, idx in mapping.items():
                raw = row[idx]
                if field_name in constants.NUMERIC_FIELDS:
                    record[field_name] = _to_number(raw)
                else:
                    record[field_name] = _clean_str(raw)

            record["feature_kind"] = feature_kind_for(
                record.get(constants.FIELD_ELEMENT_SUBTYPE)
            )
            records.append(record)

    return records
