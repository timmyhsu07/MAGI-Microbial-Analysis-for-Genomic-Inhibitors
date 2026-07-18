"""Module 2 - The Predictor.

Turns Module 1's binary AMR feature matrix into calibrated, leakage-safe,
per-drug resistance predictions with a deterministic molecular-target gate.

Read-only, defensive decision support: this package only *reads* an existing
genome's determinants and estimates a phenotype. It never designs, modifies, or
suggests changes to any organism.
"""

__version__ = "0.1.0"
