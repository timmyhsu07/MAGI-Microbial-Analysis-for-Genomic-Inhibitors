"""Decision-layer configuration.

Every threshold that shapes a call lives here, is documented, and is loadable
from YAML/JSON. **Where these are tuned:** on a held-out / validation split
(Module 2's calibration + threshold-selection work), NEVER on the test split
that the evaluation panel reports. Module 3 only *consumes* these values and
*reports* test performance; it does not fit or tune anything.

The defaults below are conservative placeholders chosen to make the no-call
policy demonstrable; a real deployment replaces them with values selected on a
validation set and records the provenance in `tuned_on`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised on invalid decision configuration."""


@dataclass(frozen=True)
class DecisionConfig:
    # --- Uncertainty band around 0.5 -------------------------------------- #
    # A calibrated probability in [band_low, band_high] is a NO_CALL (trigger a).
    # The band edges also serve as the decision boundaries: prob > band_high ->
    # resistant ("likely to fail"); prob < band_low -> susceptible ("likely to
    # work"). Widen the band to buy precision at the cost of more no-calls.
    # TUNED ON: held-out validation split (not test).
    uncertainty_band_low: float = 0.40
    uncertainty_band_high: float = 0.60

    # --- Out-of-distribution (trigger c) ---------------------------------- #
    # ood_score >= this -> NO_CALL. Scale is defined by Module 2's novelty
    # signal (0 = typical training genome, 1 = maximally novel here).
    # TUNED ON: held-out validation split.
    ood_threshold: float = 0.80

    # --- Evidence categorization ------------------------------------------ #
    # A hit counts as a "curated known mechanism" only if its detection method
    # starts with one of these (exact gene / exact allele / point mutation).
    # BLAST/PARTIAL/HMM are treated as weaker (association-tier) evidence.
    known_methods: tuple[str, ...] = ("EXACT", "POINT", "ALLELE")
    # A non-curated top feature must have |contribution| >= this to count as a
    # material statistical driver (category ii vs iii).
    assoc_min_abs_contribution: float = 0.05
    # A known-mechanism top feature must contribute >= this toward resistance to
    # count as a known-mechanism driver.
    known_feature_min_contribution: float = 0.05

    # --- Coverage --------------------------------------------------------- #
    # Species the pipeline is validated for. An input outside this list yields a
    # supported=False report (no per-drug guesses).
    covered_species: tuple[str, ...] = ("Escherichia coli",)
    # Optional explicit list of drugs of interest. Any drug here that the
    # predictor does not cover -> NO_CALL(DRUG_NOT_COVERED) (trigger d). Empty
    # means "use exactly the predictor's covered drugs".
    drugs_of_interest: tuple[str, ...] = ()

    # --- Evaluation panel ------------------------------------------------- #
    reliability_bins: int = 10

    # --- Provenance ------------------------------------------------------- #
    tuned_on: str = "placeholder defaults (replace with validation-tuned values)"

    def as_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        for k, v in raw.items():
            if isinstance(v, tuple):
                raw[k] = list(v)
        return raw


_TUPLE_KEYS = {"known_methods", "covered_species", "drugs_of_interest"}


def _coerce(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    for k in _TUPLE_KEYS:
        if k in out and out[k] is not None:
            if not isinstance(out[k], (list, tuple)):
                raise ConfigError(f"'{k}' must be a list")
            out[k] = tuple(out[k])
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> DecisionConfig:
    """Load a :class:`DecisionConfig` from YAML/JSON plus optional overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ConfigError("PyYAML required for YAML configs") from exc
            loaded = yaml.safe_load(text) or {}
        elif p.suffix == ".json":
            loaded = json.loads(text)
        else:
            raise ConfigError(f"unsupported config extension: {p.suffix}")
        if not isinstance(loaded, dict):
            raise ConfigError("config must be a mapping")
        data.update(loaded)
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    data = _coerce(data)
    valid = {item.name for item in fields(DecisionConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ConfigError(f"unknown config key(s): {sorted(unknown)}")

    cfg = DecisionConfig(**data)
    validate_config(cfg)
    return cfg


def validate_config(cfg: DecisionConfig) -> None:
    if not (0.0 <= cfg.uncertainty_band_low <= cfg.uncertainty_band_high <= 1.0):
        raise ConfigError(
            "require 0 <= uncertainty_band_low <= uncertainty_band_high <= 1; got "
            f"[{cfg.uncertainty_band_low}, {cfg.uncertainty_band_high}]"
        )
    if not (0.0 <= cfg.ood_threshold <= 1.0):
        raise ConfigError(f"ood_threshold must be in [0,1], got {cfg.ood_threshold}")
    if cfg.reliability_bins < 2:
        raise ConfigError("reliability_bins must be >= 2")
    if not cfg.known_methods:
        raise ConfigError("known_methods must not be empty")
