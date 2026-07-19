"""Report assembly (Module 3).

Orchestrates one genome's report: for each drug of interest, categorize the
evidence and apply the decision logic, then package a :class:`GenomeReport` for
the UI. Fails gracefully -- unsupported species, an unavailable predictor,
failed feature extraction, or an empty feature row each yield a clear message,
never a silent guess.
"""

from __future__ import annotations

from pathlib import Path

from .catalog import classes_for
from .config import DecisionConfig
from .contracts import (
    FeatureBundle,
    FeatureExtractionError,
    FeatureExtractor,
    GenomeReport,
    PredictorUnavailable,
    Predictor,
)
from .decision import decide, no_call_uncovered
from .evidence import categorize_evidence

MANDATORY_DISCLAIMER = (
    "DECISION SUPPORT ONLY - every result must be confirmed by standard "
    "laboratory antimicrobial susceptibility testing. This tool does not make "
    "treatment decisions and requires trained human oversight."
)


def build_report(
    features: FeatureBundle,
    predictor: Predictor,
    config: DecisionConfig,
    species: str,
) -> GenomeReport:
    """Build a per-genome report from already-extracted features + a predictor."""
    covered = predictor.covered_drugs()
    drugs = list(config.drugs_of_interest) if config.drugs_of_interest else list(covered)
    uncovered_requested = [d for d in drugs if d not in covered]

    # Graceful failure: unsupported species -> no per-drug guesses.
    if species not in config.covered_species:
        return GenomeReport(
            genome_id=features.genome_id,
            species=species,
            species_supported=False,
            decisions=[],
            covered_drugs=covered,
            uncovered_drugs_requested=uncovered_requested,
            disclaimer=MANDATORY_DISCLAIMER,
            errors=[
                f"Species '{species}' is not covered by this pipeline "
                f"(covered: {list(config.covered_species)}). No predictions made."
            ],
        )

    # Graceful failure: predictor artifact/service unavailable.
    try:
        predictions = predictor.predict(features)
    except PredictorUnavailable as exc:
        return GenomeReport(
            genome_id=features.genome_id,
            species=species,
            species_supported=True,
            decisions=[],
            covered_drugs=covered,
            uncovered_drugs_requested=uncovered_requested,
            disclaimer=MANDATORY_DISCLAIMER,
            errors=[f"Predictor unavailable: {exc}. No predictions made."],
        )

    pred_by_drug = {p.drug: p for p in predictions}
    errors: list[str] = []
    if features.is_empty:
        errors.append(
            "No features were extracted for this genome: every prediction is "
            "driven purely by the ABSENCE of markers. Interpret with extra "
            "caution and confirm by laboratory testing."
        )

    decisions = []
    for drug in drugs:
        if drug not in covered or drug not in pred_by_drug:
            decisions.append(no_call_uncovered(drug))
            continue
        prediction = pred_by_drug[drug]
        evidence = categorize_evidence(prediction, features, classes_for(drug), config)
        decisions.append(decide(prediction, evidence, config))

    return GenomeReport(
        genome_id=features.genome_id,
        species=species,
        species_supported=True,
        decisions=decisions,
        covered_drugs=covered,
        uncovered_drugs_requested=uncovered_requested,
        disclaimer=MANDATORY_DISCLAIMER,
        errors=errors,
    )


def report_from_fasta(
    fasta_path: str,
    feature_extractor: FeatureExtractor,
    predictor: Predictor,
    config: DecisionConfig,
    species: str,
) -> GenomeReport:
    """Extract features from a FASTA then build the report. Extraction failure is
    reported, not raised."""
    try:
        features = feature_extractor(fasta_path)
    except FeatureExtractionError as exc:
        return GenomeReport(
            genome_id=Path(fasta_path).name,
            species=species,
            species_supported=species in config.covered_species,
            decisions=[],
            covered_drugs=predictor.covered_drugs(),
            disclaimer=MANDATORY_DISCLAIMER,
            errors=[f"Feature extraction failed: {exc}"],
        )
    return build_report(features, predictor, config, species)
