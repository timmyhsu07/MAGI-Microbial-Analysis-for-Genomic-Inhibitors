"""Module 3 - The Decision Report.

Turns Module 2's per-drug predictions (plus Module 1's provenance) into a
human-facing, per-drug decision card: a label (likely to fail / likely to work /
no-call), an evidence category, and a rationale.

Read-only, defensive decision support: this package only *interprets* existing
predictions and hands them to a human for confirmation by laboratory testing.
It never makes a treatment decision and never designs, modifies, or suggests
changes to any organism.
"""

__version__ = "0.1.0"
