"""Classify prediction evidence as known mechanism, association, or no signal.

The category combines Module 1 hits with Module 2 feature contributions and is
computed independently from the final decision label.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import DecisionConfig
from .contracts import (
    DrugPrediction,
    EvidenceCategory,
    FeatureBundle,
    LongHit,
    TopFeature,
)


@dataclass(frozen=True)
class EvidenceResult:
    category: EvidenceCategory
    curated_hits: list[LongHit]  # exact known-mechanism hits for this drug class
    associated_hits: list[LongHit]  # right-class hits seen only by weak methods
    driver_features: list[TopFeature]  # material non-curated statistical drivers


def method_is_curated(method: str | None, known_prefixes: tuple[str, ...]) -> bool:
    """True if a detection method denotes a curated exact call."""
    if not method:
        return False
    m = method.strip().upper()
    return any(m.startswith(p.upper()) for p in known_prefixes)


def categorize_evidence(
    prediction: DrugPrediction,
    features: FeatureBundle,
    drug_classes: set[str],
    config: DecisionConfig,
) -> EvidenceResult:
    """Categorize the evidence for one drug prediction.

    ``drug_classes`` is the set of AMR classes this drug belongs to (from the
    drug catalog), used to match Module 1 hits by ``drug_class``.
    """
    classes_upper = {c.upper() for c in drug_classes}

    curated_hits: list[LongHit] = []
    associated_hits: list[LongHit] = []
    for hit in features.long_hits:
        if (hit.element_type or "").upper() != "AMR":
            continue
        if not hit.drug_class or hit.drug_class.upper() not in classes_upper:
            continue
        if method_is_curated(hit.method, config.known_methods):
            curated_hits.append(hit)
        else:
            associated_hits.append(hit)

    known_feature_driver = any(
        f.is_known_mechanism and f.contribution >= config.known_feature_min_contribution
        for f in prediction.top_features
    )
    driver_features = [
        f
        for f in prediction.top_features
        if not f.is_known_mechanism
        and abs(f.contribution) >= config.assoc_min_abs_contribution
    ]

    if curated_hits or known_feature_driver:
        category = EvidenceCategory.KNOWN_MECHANISM
    elif associated_hits or driver_features:
        category = EvidenceCategory.ASSOCIATION_ONLY
    else:
        category = EvidenceCategory.NO_SIGNAL

    return EvidenceResult(
        category=category,
        curated_hits=curated_hits,
        associated_hits=associated_hits,
        driver_features=driver_features,
    )
