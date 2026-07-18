"""Deterministic target-gene gate.

The gate is deliberately narrower than the statistical model. It only fires
when the target state is known for every configured target and all are absent;
unknown target state is left for the model path and audit log.
"""

from __future__ import annotations

from collections.abc import Mapping


def apply_gate(
    target_row: Mapping[str, int], target_genes: list[str], on_target_absent: str
) -> str | None:
    for gene in target_genes:
        if gene not in target_row:
            return None
        if int(target_row[gene]) != 0:
            return None
    return on_target_absent
