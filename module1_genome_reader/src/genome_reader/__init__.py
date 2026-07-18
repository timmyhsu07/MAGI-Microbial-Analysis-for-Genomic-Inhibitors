"""Module 1 - The Genome Reader.

Turns a directory of assembled bacterial genome FASTA files into a documented,
reproducible ML feature matrix using AMRFinderPlus. Defensive resistance
detection only: this package annotates and tabulates existing resistance
determinants; it never designs, modifies, or suggests changes to an organism.
"""

from __future__ import annotations

from .config import Config, ConfigError, load_config
from .constants import PIPELINE_VERSION, SCHEMA_VERSION
from .pipeline import PipelineError, RunResult, run_pipeline

__all__ = [
    "Config",
    "ConfigError",
    "load_config",
    "run_pipeline",
    "RunResult",
    "PipelineError",
    "PIPELINE_VERSION",
    "SCHEMA_VERSION",
]

__version__ = PIPELINE_VERSION
