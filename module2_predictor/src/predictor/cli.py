"""Command-line entrypoint for the predictor."""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config
from .distance import DistanceError
from .io import InputError
from .train import TrainingError, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predictor",
        description="Module 2 - leakage-safe AMR phenotype predictor.",
    )
    p.add_argument("--config", required=True, help="Path to a YAML/JSON config file.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        result = run(cfg)
    except ConfigError as exc:
        print(f"ERROR (config): {exc}", file=sys.stderr)
        return 2
    except InputError as exc:
        print(f"ERROR (input): {exc}", file=sys.stderr)
        return 3
    except DistanceError as exc:
        print(f"ERROR (distance): {exc}", file=sys.stderr)
        return 4
    except TrainingError as exc:
        print(f"ERROR (training): {exc}", file=sys.stderr)
        return 5

    print(
        f"OK: trained {len(result.trained_drugs)} drug model(s) over "
        f"{result.n_genomes} genomes / {result.n_clusters} clusters\n"
        f"  {result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
