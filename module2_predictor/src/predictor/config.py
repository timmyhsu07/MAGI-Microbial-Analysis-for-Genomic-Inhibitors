"""Configuration loading for the predictor.

The predictor consumes artifacts produced elsewhere, so path handling is the
main source of preventable mistakes. Input paths are resolved against the config
file that names them; output paths are left as the caller provided so CLI runs
from a project directory behave naturally.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised on malformed or invalid configuration."""


@dataclass(frozen=True)
class InputPaths:
    feature_matrix: str
    feature_schema: str
    labels: str
    target_gene_table: str


@dataclass(frozen=True)
class DistanceConfig:
    method: str
    mash_sketch_dir: str | None
    cluster_threshold: float


@dataclass(frozen=True)
class CVConfig:
    n_splits: int
    calibration_fraction: float
    min_per_class: int


@dataclass(frozen=True)
class ModelConfig:
    C: float
    max_iter: int


@dataclass(frozen=True)
class OODConfig:
    threshold: float


@dataclass(frozen=True)
class DecisionConfig:
    min_confidence: float


@dataclass(frozen=True)
class Config:
    seed: int
    output_dir: str
    inputs: InputPaths
    distance: DistanceConfig
    cv: CVConfig
    model: ModelConfig
    ood: OODConfig
    decision: DecisionConfig
    drugs: dict[str, dict[str, Any]]

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)


def _load_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigError(
                f"{path}: PyYAML is required to read YAML config files"
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


def _resolve_input_paths(inputs: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    out = dict(inputs)
    for key, value in out.items():
        if value is None:
            continue
        p = Path(value)
        out[key] = str(p if p.is_absolute() else (base_dir / p).resolve())
    return out


def _build_drugs(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    drugs: dict[str, dict[str, Any]] = {}
    for drug, data in raw.items():
        if not isinstance(data, dict):
            raise ConfigError(f"drugs.{drug} must be a mapping")
        target_genes = data.get("target_genes")
        if not isinstance(target_genes, list) or not all(
            isinstance(g, str) for g in target_genes
        ):
            raise ConfigError(f"drugs.{drug}.target_genes must be a list of strings")
        forced = data.get("on_target_absent")
        if forced not in {"resistant", "susceptible", "no_call"}:
            raise ConfigError(
                f"drugs.{drug}.on_target_absent must be resistant, susceptible, or no_call"
            )
        drugs[str(drug)] = {
            "target_genes": list(target_genes),
            "on_target_absent": str(forced),
        }
    return drugs


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load the reference config and apply top-level overrides."""
    config_path = Path(path).resolve()
    data = _load_file(config_path)
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    required = {
        "seed",
        "output_dir",
        "inputs",
        "distance",
        "cv",
        "model",
        "ood",
        "decision",
        "drugs",
    }
    missing = required - set(data)
    if missing:
        raise ConfigError(f"Missing required config key(s): {sorted(missing)}")
    unknown = set(data) - required
    if unknown:
        raise ConfigError(f"Unknown config key(s): {sorted(unknown)}")

    inputs = InputPaths(**_resolve_input_paths(data["inputs"], config_path.parent))
    cfg = Config(
        seed=int(data["seed"]),
        output_dir=str(data["output_dir"]),
        inputs=inputs,
        distance=DistanceConfig(**data["distance"]),
        cv=CVConfig(**data["cv"]),
        model=ModelConfig(**data["model"]),
        ood=OODConfig(**data["ood"]),
        decision=DecisionConfig(**data["decision"]),
        drugs=_build_drugs(data["drugs"]),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if cfg.distance.method not in {"jaccard", "mash"}:
        raise ConfigError("distance.method must be 'jaccard' or 'mash'")
    if cfg.distance.cluster_threshold < 0:
        raise ConfigError("distance.cluster_threshold must be >= 0")
    if cfg.cv.n_splits < 1:
        raise ConfigError("cv.n_splits must be >= 1")
    if not 0 <= cfg.cv.calibration_fraction < 1:
        raise ConfigError("cv.calibration_fraction must be in [0, 1)")
    if cfg.cv.min_per_class < 1:
        raise ConfigError("cv.min_per_class must be >= 1")
    if cfg.model.C <= 0:
        raise ConfigError("model.C must be > 0")
    if cfg.model.max_iter < 1:
        raise ConfigError("model.max_iter must be >= 1")
    if not 0.5 <= cfg.decision.min_confidence <= 1.0:
        raise ConfigError("decision.min_confidence must be in [0.5, 1.0]")

    for key, value in cfg.inputs.__dict__.items():
        if not Path(value).exists():
            raise ConfigError(f"inputs.{key} does not exist: {value}")
