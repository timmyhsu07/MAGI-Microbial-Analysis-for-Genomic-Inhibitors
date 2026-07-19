"""Run configuration for the Genome Reader.

Everything the pipeline needs is captured in a single :class:`Config` object.
Values can come from a YAML or JSON file and/or be overridden individually (the
CLI layer passes overrides straight through). The config is validated eagerly
so the run fails *before* any genome is annotated if, for example, the organism
is unknown or the input directory is missing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from . import constants


class ConfigError(ValueError):
    """Raised on malformed or invalid configuration."""


@dataclass(frozen=True)
class Config:
    # --- I/O -------------------------------------------------------------- #
    input_dir: str
    output_dir: str

    # --- AMRFinderPlus ---------------------------------------------------- #
    # Organism enables point-mutation screening. ``None`` selects acquired-gene
    # only mode (no point mutations). If set, it must be a recognized
    # AMRFinderPlus organism value.
    organism: str | None = None
    amrfinder_bin: str = "amrfinder"
    # Path to the AMRFinderPlus database directory (the ".../latest" dir). When
    # None, AMRFinderPlus uses the database bundled with the install.
    database_dir: str | None = None
    # Include stress/virulence screening (adds the --plus flag). This only
    # affects the long/provenance table unless those element types are also
    # added to ``include_element_types`` below.
    use_plus: bool = True
    # Threads AMRFinderPlus itself uses per genome.
    amrfinder_threads: int = 1

    # --- Parallelism / caching ------------------------------------------- #
    workers: int = 4  # genomes annotated concurrently
    reuse_cache: bool = True
    cache_dir: str | None = None  # defaults to <output_dir>/cache
    # When False (default) the run fails loudly if ANY genome fails annotation.
    # Set True to emit a matrix from the genomes that succeeded and record the
    # failures in the manifest instead of aborting.
    allow_partial: bool = False

    # --- Discovery -------------------------------------------------------- #
    nucleotide_extensions: tuple[str, ...] = constants.DEFAULT_NUCLEOTIDE_EXTENSIONS
    protein_extensions: tuple[str, ...] = constants.DEFAULT_PROTEIN_EXTENSIONS
    # What to do with files detected as protein: "skip" (log & exclude) or
    # "annotate" (run AMRFinderPlus in -p protein mode; no point mutations).
    protein_handling: str = "skip"

    # --- Feature matrix --------------------------------------------------- #
    # Element types (AMRFinderPlus "Type" column) that become feature columns
    # in the binary matrix. The long/provenance table always retains ALL hits
    # regardless of this setting.
    include_element_types: tuple[str, ...] = ("AMR",)
    # How an unseen element symbol at inference time is handled. This is a
    # documented contract, surfaced in the schema; the value is recorded, not
    # enforced by Module 1 (which only builds the corpus matrix).
    unseen_feature_policy: str = "dropped_and_logged"

    # --- Run metadata ----------------------------------------------------- #
    run_id: str | None = None  # filled in by the pipeline if None

    def resolved_cache_dir(self) -> Path:
        if self.cache_dir:
            return Path(self.cache_dir)
        return Path(self.output_dir) / "cache"

    def to_serializable(self) -> dict[str, Any]:
        """Config as a plain JSON-serializable dict (tuples -> lists)."""
        raw = asdict(self)
        for key, value in raw.items():
            if isinstance(value, tuple):
                raw[key] = list(value)
        return raw


# Keys that must be provided as tuples internally but may arrive as lists.
_TUPLE_KEYS = {
    "nucleotide_extensions",
    "protein_extensions",
    "include_element_types",
}

_VALID_PROTEIN_HANDLING = {"skip", "annotate"}


def _coerce_types(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    for key in _TUPLE_KEYS:
        if key in out and out[key] is not None:
            if not isinstance(out[key], (list, tuple)):
                raise ConfigError(f"'{key}' must be a list, got {type(out[key]).__name__}")
            out[key] = tuple(out[key])
    return out


def _load_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # imported lazily so JSON configs need no PyYAML
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigError(
                f"{path}: PyYAML is required to read YAML config files. "
                "Install pyyaml or use a .json config."
            ) from exc
        loaded = yaml.safe_load(text)
    elif path.suffix == ".json":
        loaded = json.loads(text)
    else:
        raise ConfigError(
            f"{path}: unsupported config extension '{path.suffix}' "
            "(use .yaml, .yml, or .json)"
        )
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top-level config must be a mapping/object")
    return loaded


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Build a validated :class:`Config` from an optional file plus overrides.

    ``overrides`` (typically from CLI flags) win over file values. Unknown keys
    fail loudly so a typo in a config file can never be silently ignored.
    """
    data: dict[str, Any] = {}
    if config_path is not None:
        data.update(_load_file(Path(config_path)))
    if overrides:
        # Drop overrides that are None so they don't clobber file values.
        data.update({k: v for k, v in overrides.items() if v is not None})

    data = _coerce_types(data)

    valid_keys = {item.name for item in fields(Config)}
    unknown = set(data) - valid_keys
    if unknown:
        raise ConfigError(
            f"Unknown config key(s): {sorted(unknown)}. Valid keys: {sorted(valid_keys)}"
        )
    if "input_dir" not in data or "output_dir" not in data:
        missing = [k for k in ("input_dir", "output_dir") if k not in data]
        raise ConfigError(f"Missing required config key(s): {missing}")

    cfg = Config(**data)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    """Raise :class:`ConfigError` on any invalid setting."""
    in_dir = Path(cfg.input_dir)
    if not in_dir.exists():
        raise ConfigError(f"input_dir does not exist: {cfg.input_dir}")
    if not in_dir.is_dir():
        raise ConfigError(f"input_dir is not a directory: {cfg.input_dir}")

    if cfg.organism is not None and cfg.organism not in constants.VALID_ORGANISMS:
        raise ConfigError(
            f"Unknown organism '{cfg.organism}'. Point-mutation screening "
            "requires a recognized AMRFinderPlus organism. Valid values: "
            f"{sorted(constants.VALID_ORGANISMS)}"
        )

    if cfg.workers < 1:
        raise ConfigError(f"workers must be >= 1, got {cfg.workers}")
    if cfg.amrfinder_threads < 1:
        raise ConfigError(f"amrfinder_threads must be >= 1, got {cfg.amrfinder_threads}")

    if cfg.protein_handling not in _VALID_PROTEIN_HANDLING:
        raise ConfigError(
            f"protein_handling must be one of {sorted(_VALID_PROTEIN_HANDLING)}, "
            f"got '{cfg.protein_handling}'"
        )

    if cfg.database_dir is not None and not Path(cfg.database_dir).exists():
        raise ConfigError(
            f"database_dir does not exist: {cfg.database_dir}. Run the setup "
            "script to download the AMRFinderPlus database, or omit database_dir "
            "to use the install's bundled database."
        )

    if not cfg.include_element_types:
        raise ConfigError("include_element_types must not be empty")


def with_run_id(cfg: Config, run_id: str) -> Config:
    """Return a copy of ``cfg`` with ``run_id`` set."""
    return replace(cfg, run_id=run_id)
