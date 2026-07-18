"""Constants for the Genome Reader (Module 1).

This module is the single source of truth for:

* schema / pipeline versions,
* the mapping from AMRFinderPlus TSV headers (which changed between the 3.x and
  4.x releases) onto the pipeline's stable, canonical field names, and
* the set of recognized ``--organism`` values.

Keeping the header-alias table here means the parser tolerates both the current
4.x column names and the older 3.x names without any code changes elsewhere.
Verified against the NCBI AMRFinderPlus wiki (Running-AMRFinderPlus) in
July 2026.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Versions
# --------------------------------------------------------------------------- #

# Version of THIS pipeline's code.
PIPELINE_VERSION = "0.1.0"

# Version of the emitted feature_schema.json / matrix layout contract.
# Bump the MAJOR component whenever a change would break a downstream reader
# (e.g. renaming/removing a column or changing the value encoding).
SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Canonical parsed-hit fields
# --------------------------------------------------------------------------- #
# These are the stable internal names. Everything downstream (long table,
# schema, matrix) is written in terms of these, never the raw TSV headers.

FIELD_ELEMENT_SYMBOL = "element_symbol"
FIELD_ELEMENT_NAME = "element_name"
FIELD_SCOPE = "scope"
FIELD_ELEMENT_TYPE = "element_type"
FIELD_ELEMENT_SUBTYPE = "element_subtype"
FIELD_CLASS = "class"
FIELD_SUBCLASS = "subclass"
FIELD_METHOD = "method"
FIELD_PCT_IDENTITY = "pct_identity"
FIELD_PCT_COVERAGE = "pct_coverage"
FIELD_CONTIG_ID = "contig_id"
FIELD_START = "start"
FIELD_STOP = "stop"
FIELD_STRAND = "strand"
FIELD_PROTEIN_ID = "protein_id"
FIELD_TARGET_LENGTH = "target_length"
FIELD_REF_LENGTH = "reference_length"
FIELD_ALIGNMENT_LENGTH = "alignment_length"
FIELD_CLOSEST_ACCESSION = "closest_reference_accession"
FIELD_CLOSEST_NAME = "closest_reference_name"
FIELD_HMM_ACCESSION = "hmm_accession"
FIELD_HMM_DESCRIPTION = "hmm_description"

# Fields we require to build the feature matrix. If, after alias resolution, any
# of these is missing from a TSV, the parser fails loudly rather than silently
# producing an empty/wrong matrix.
REQUIRED_FIELDS = (
    FIELD_ELEMENT_SYMBOL,
    FIELD_ELEMENT_TYPE,
    FIELD_ELEMENT_SUBTYPE,
    FIELD_METHOD,
)

# --------------------------------------------------------------------------- #
# Header alias table: canonical field -> accepted TSV header spellings.
# The FIRST spelling in each list is the current (4.x) name; later entries are
# older (3.x) names retained for backward compatibility. Header matching is
# done case-insensitively after stripping surrounding whitespace.
# --------------------------------------------------------------------------- #

HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    FIELD_PROTEIN_ID: ("Protein id", "Protein identifier"),
    FIELD_CONTIG_ID: ("Contig id",),
    FIELD_START: ("Start",),
    FIELD_STOP: ("Stop",),
    FIELD_STRAND: ("Strand",),
    FIELD_ELEMENT_SYMBOL: ("Element symbol", "Gene symbol"),
    FIELD_ELEMENT_NAME: ("Element name", "Sequence name"),
    FIELD_SCOPE: ("Scope",),
    FIELD_ELEMENT_TYPE: ("Type", "Element type"),
    FIELD_ELEMENT_SUBTYPE: ("Subtype", "Element subtype"),
    FIELD_CLASS: ("Class",),
    FIELD_SUBCLASS: ("Subclass",),
    FIELD_METHOD: ("Method",),
    FIELD_TARGET_LENGTH: ("Target length",),
    FIELD_REF_LENGTH: ("Reference sequence length",),
    FIELD_PCT_COVERAGE: (
        "% Coverage of reference",
        "% Coverage of reference sequence",
    ),
    FIELD_PCT_IDENTITY: (
        "% Identity to reference",
        "% Identity to reference sequence",
    ),
    FIELD_ALIGNMENT_LENGTH: ("Alignment length",),
    FIELD_CLOSEST_ACCESSION: (
        "Closest reference accession",
        "Accession of closest sequence",
    ),
    FIELD_CLOSEST_NAME: (
        "Closest reference name",
        "Name of closest sequence",
    ),
    FIELD_HMM_ACCESSION: ("HMM accession", "HMM id"),
    FIELD_HMM_DESCRIPTION: ("HMM description",),
}

# Numeric fields (parsed to float; blank/NA -> None).
NUMERIC_FIELDS = frozenset(
    {
        FIELD_PCT_IDENTITY,
        FIELD_PCT_COVERAGE,
        FIELD_START,
        FIELD_STOP,
        FIELD_TARGET_LENGTH,
        FIELD_REF_LENGTH,
        FIELD_ALIGNMENT_LENGTH,
    }
)

# Column order for the emitted long-format provenance table.
LONG_TABLE_COLUMNS = (
    "genome_id",
    FIELD_ELEMENT_SYMBOL,
    FIELD_ELEMENT_NAME,
    FIELD_SCOPE,
    FIELD_ELEMENT_TYPE,
    FIELD_ELEMENT_SUBTYPE,
    "feature_kind",
    FIELD_CLASS,
    FIELD_SUBCLASS,
    FIELD_METHOD,
    FIELD_PCT_IDENTITY,
    FIELD_PCT_COVERAGE,
    FIELD_CONTIG_ID,
    FIELD_START,
    FIELD_STOP,
    FIELD_STRAND,
    FIELD_PROTEIN_ID,
    FIELD_TARGET_LENGTH,
    FIELD_REF_LENGTH,
    FIELD_ALIGNMENT_LENGTH,
    FIELD_CLOSEST_ACCESSION,
    FIELD_CLOSEST_NAME,
    FIELD_HMM_ACCESSION,
    FIELD_HMM_DESCRIPTION,
)


# --------------------------------------------------------------------------- #
# AMRFinderPlus element vocabulary
# --------------------------------------------------------------------------- #

# Element subtype used by AMRFinderPlus to flag point mutations. Everything
# else (chiefly "AMR") is treated as an acquired element for feature-kind
# purposes.
SUBTYPE_POINT = "POINT"

FEATURE_KIND_ACQUIRED = "acquired_gene"
FEATURE_KIND_POINT = "point_mutation"

# Documented Method values (informational; used only for validation warnings).
# Base methods carry a 'P' (protein) or 'X' (translated nucleotide) suffix in
# practice, e.g. EXACTX, BLASTP, POINTX.
KNOWN_METHOD_BASES = frozenset(
    {
        "ALLELE",
        "EXACT",
        "BLAST",
        "PARTIAL",
        "PARTIAL_CONTIG_END",
        "HMM",
        "INTERNAL_STOP",
        "POINT",
    }
)


# --------------------------------------------------------------------------- #
# Recognized --organism values (AMRFinderPlus, verified July 2026).
# Point-mutation screening requires one of these. Kept here so the config layer
# can validate the organism before any genome is annotated.
# --------------------------------------------------------------------------- #

VALID_ORGANISMS = frozenset(
    {
        "Acinetobacter_baumannii",
        "Bordetella_pertussis",
        "Burkholderia_cepacia",
        "Burkholderia_mallei",
        "Burkholderia_pseudomallei",
        "Campylobacter",
        "Citrobacter_freundii",
        "Clostridioides_difficile",
        "Corynebacterium_diphtheriae",
        "Enterobacter_asburiae",
        "Enterobacter_cloacae",
        "Enterococcus_faecalis",
        "Enterococcus_faecium",
        "Escherichia",
        "Haemophilus_influenzae",
        "Klebsiella_oxytoca",
        "Klebsiella_pneumoniae",
        "Neisseria_gonorrhoeae",
        "Neisseria_meningitidis",
        "Pseudomonas_aeruginosa",
        "Salmonella",
        "Serratia_marcescens",
        "Staphylococcus_aureus",
        "Staphylococcus_epidermidis",
        "Staphylococcus_pseudintermedius",
        "Streptococcus_agalactiae",
        "Streptococcus_pneumoniae",
        "Streptococcus_pyogenes",
        "Vibrio_cholerae",
        "Vibrio_parahaemolyticus",
        "Vibrio_vulnificus",
    }
)


# --------------------------------------------------------------------------- #
# File-discovery defaults
# --------------------------------------------------------------------------- #

DEFAULT_NUCLEOTIDE_EXTENSIONS = (".fasta", ".fa", ".fna", ".fas", ".seq")
DEFAULT_PROTEIN_EXTENSIONS = (".faa", ".fpaa")

# Output artifact filenames.
OUT_BINARY_PARQUET = "features_binary.parquet"
OUT_BINARY_CSV = "features_binary.csv"
OUT_LONG_PARQUET = "features_long.parquet"
OUT_SCHEMA_JSON = "feature_schema.json"
OUT_MANIFEST_JSON = "run_manifest.json"

ID_COLUMN = "genome_id"
