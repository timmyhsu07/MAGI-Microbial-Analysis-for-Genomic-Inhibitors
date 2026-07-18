"""Stage 5 - Versioned column manifest (feature_schema.json).

Describes every feature column in the binary matrix so a downstream consumer
can interpret the matrix without re-reading AMRFinderPlus output: for each
column we record the element symbol, whether it is an acquired gene or a point
mutation, the AMRFinderPlus element type/subtype, associated drug class(es),
the methods that produced it, and how many genomes carried it.

The top level also records the matrix contract (id column, value encoding,
unseen-feature policy) and run provenance (organism, AMRFinderPlus software +
database versions), so the schema is a self-contained description of the run.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from . import constants
from .config import Config
from .parse import feature_kind_for


def _sorted_unique(values: set[str]) -> list[str]:
    return sorted(v for v in values if v)


def build_schema(
    records: list[dict[str, Any]],
    feature_columns: list[str],
    matrix_genome_ids: list[str],
    cfg: Config,
    versions: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the feature_schema.json structure.

    ``feature_columns`` is the authoritative, ordered column list from the
    matrix builder; this function only annotates those columns (it never adds
    or drops one), so the schema and the matrix can never disagree.
    """
    include = {t.upper() for t in cfg.include_element_types}
    matrix_gids = set(matrix_genome_ids)

    # Aggregate per-symbol metadata across qualifying hits.
    subtypes: dict[str, set[str]] = defaultdict(set)
    types: dict[str, set[str]] = defaultdict(set)
    classes: dict[str, set[str]] = defaultdict(set)
    subclasses: dict[str, set[str]] = defaultdict(set)
    methods: dict[str, set[str]] = defaultdict(set)
    genomes_present: dict[str, set[str]] = defaultdict(set)

    for rec in records:
        symbol = rec.get(constants.FIELD_ELEMENT_SYMBOL)
        etype = (rec.get(constants.FIELD_ELEMENT_TYPE) or "").upper()
        if not symbol or etype not in include:
            continue
        types[symbol].add(rec.get(constants.FIELD_ELEMENT_TYPE) or "")
        if rec.get(constants.FIELD_ELEMENT_SUBTYPE):
            subtypes[symbol].add(rec[constants.FIELD_ELEMENT_SUBTYPE])
        if rec.get(constants.FIELD_CLASS):
            classes[symbol].add(rec[constants.FIELD_CLASS])
        if rec.get(constants.FIELD_SUBCLASS):
            subclasses[symbol].add(rec[constants.FIELD_SUBCLASS])
        if rec.get(constants.FIELD_METHOD):
            methods[symbol].add(rec[constants.FIELD_METHOD])
        gid = rec.get("genome_id")
        if gid in matrix_gids:
            genomes_present[symbol].add(gid)

    columns = []
    for symbol in feature_columns:  # already sorted by the matrix builder
        symbol_subtypes = subtypes.get(symbol, set())
        # feature_kind is POINT if any observed subtype is POINT, else acquired.
        kind = constants.FEATURE_KIND_ACQUIRED
        for st in symbol_subtypes:
            if feature_kind_for(st) == constants.FEATURE_KIND_POINT:
                kind = constants.FEATURE_KIND_POINT
                break
        type_list = _sorted_unique(types.get(symbol, set()))
        columns.append(
            {
                "column": symbol,
                "element_symbol": symbol,
                "feature_kind": kind,
                "element_type": type_list[0] if type_list else None,
                "element_types_observed": type_list,
                "element_subtypes_observed": _sorted_unique(symbol_subtypes),
                "drug_classes": _sorted_unique(classes.get(symbol, set())),
                "drug_subclasses": _sorted_unique(subclasses.get(symbol, set())),
                "methods_observed": _sorted_unique(methods.get(symbol, set())),
                "n_genomes_present": len(genomes_present.get(symbol, set())),
            }
        )

    amr = versions.get("amrfinderplus", {})
    return {
        "schema_version": constants.SCHEMA_VERSION,
        "pipeline_version": constants.PIPELINE_VERSION,
        "generated_at": generated_at,
        "matrix": {
            "id_column": constants.ID_COLUMN,
            "value_encoding": "presence_absence",
            "value_domain": [0, 1],
            "dtype": "int8",
            "n_genomes": len(matrix_genome_ids),
            "n_features": len(feature_columns),
            "included_element_types": list(cfg.include_element_types),
            "column_order": "ascending lexicographic by element_symbol",
            "unseen_feature_policy": cfg.unseen_feature_policy,
            "unseen_feature_policy_detail": (
                "At inference time the feature columns are FIXED to this "
                "manifest. A gene/mutation not present here is dropped from the "
                "feature vector and logged; the matrix column set never grows "
                "from unseen genomes. A known column absent in a new genome is "
                "encoded 0."
            ),
        },
        "provenance": {
            "organism": cfg.organism,
            "use_plus": cfg.use_plus,
            "amrfinderplus_software_version": amr.get("software_version"),
            "amrfinderplus_database_version": amr.get("database_version"),
        },
        "columns": columns,
    }
