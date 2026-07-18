"""Command-line entrypoint: ``python -m genome_reader``.

Config precedence: built-in defaults < config file (--config) < individual CLI
flags. Any flag left unset does not override the file/default value.
"""

from __future__ import annotations

import argparse
import sys

from . import constants
from .config import ConfigError, load_config
from .pipeline import PipelineError, run_pipeline
from .versions import VersionError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="genome_reader",
        description=(
            "Module 1 - The Genome Reader: assembled bacterial genome FASTAs -> "
            "reproducible AMR feature matrix via AMRFinderPlus."
        ),
    )
    p.add_argument("--config", help="Path to a YAML/JSON config file.")
    p.add_argument("--input-dir", dest="input_dir", help="Directory of genome FASTA files.")
    p.add_argument("--output-dir", dest="output_dir", help="Output directory for artifacts.")
    p.add_argument(
        "--organism",
        help="AMRFinderPlus --organism value (enables point-mutation screening).",
    )
    p.add_argument("--amrfinder-bin", dest="amrfinder_bin", help="Path to the amrfinder binary.")
    p.add_argument("--database-dir", dest="database_dir", help="AMRFinderPlus database directory (…/latest).")
    p.add_argument("--cache-dir", dest="cache_dir", help="Cache directory (default <output_dir>/cache).")
    p.add_argument("--workers", type=int, help="Genomes annotated concurrently.")
    p.add_argument("--amrfinder-threads", dest="amrfinder_threads", type=int, help="Threads per AMRFinderPlus run.")
    p.add_argument(
        "--protein-handling",
        dest="protein_handling",
        choices=["skip", "annotate"],
        help="What to do with protein FASTAs (default: skip).",
    )
    p.add_argument(
        "--include-element-types",
        dest="include_element_types",
        help="Comma-separated element types that become matrix columns (default: AMR).",
    )

    plus = p.add_mutually_exclusive_group()
    plus.add_argument("--plus", dest="use_plus", action="store_true", default=None,
                      help="Enable AMRFinderPlus --plus (stress/virulence).")
    plus.add_argument("--no-plus", dest="use_plus", action="store_false", default=None,
                      help="Disable --plus.")

    cache = p.add_mutually_exclusive_group()
    cache.add_argument("--cache", dest="reuse_cache", action="store_true", default=None,
                       help="Reuse cached annotations (default).")
    cache.add_argument("--no-cache", dest="reuse_cache", action="store_false", default=None,
                       help="Ignore cache and re-annotate every genome.")

    p.add_argument("--allow-partial", dest="allow_partial", action="store_true", default=None,
                   help="Emit a matrix even if some genomes fail annotation.")
    p.add_argument(
        "--skip-tool-check",
        dest="skip_tool_check",
        action="store_true",
        help="Do not require a resolvable AMRFinderPlus version (e.g. when "
        "running purely from a pre-populated cache).",
    )
    p.add_argument(
        "--list-organisms",
        action="store_true",
        help="Print the recognized AMRFinderPlus organism values and exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_organisms:
        for org in sorted(constants.VALID_ORGANISMS):
            print(org)
        return 0

    overrides = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "organism": args.organism,
        "amrfinder_bin": args.amrfinder_bin,
        "database_dir": args.database_dir,
        "cache_dir": args.cache_dir,
        "workers": args.workers,
        "amrfinder_threads": args.amrfinder_threads,
        "protein_handling": args.protein_handling,
        "use_plus": args.use_plus,
        "reuse_cache": args.reuse_cache,
        "allow_partial": args.allow_partial,
    }
    if args.include_element_types is not None:
        overrides["include_element_types"] = tuple(
            t.strip() for t in args.include_element_types.split(",") if t.strip()
        )

    try:
        cfg = load_config(args.config, overrides)
    except ConfigError as exc:
        print(f"ERROR (config): {exc}", file=sys.stderr)
        return 2

    try:
        result = run_pipeline(cfg, require_amrfinder=not args.skip_tool_check)
    except VersionError as exc:
        print(f"ERROR (versions): {exc}", file=sys.stderr)
        return 3
    except PipelineError as exc:
        print(f"ERROR (pipeline): {exc}", file=sys.stderr)
        return 4

    print(
        f"OK: run {result.run_id} - {result.n_genomes_in_matrix} genomes x "
        f"{result.n_features} features ({result.n_hits} hits)\n"
        f"  {result.binary_matrix_path}\n"
        f"  {result.long_table_path}\n"
        f"  {result.schema_path}\n"
        f"  {result.manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
