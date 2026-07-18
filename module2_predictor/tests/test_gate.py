"""The deterministic target gate overrides the model; check exactly when."""

from __future__ import annotations

from predictor.gate import apply_gate


def test_fires_when_every_target_is_absent():
    # No molecular target detected -> the drug's call is fixed, model ignored.
    assert apply_gate({"gyrA": 0, "parC": 0}, ["gyrA", "parC"], "no_call") == "no_call"
    assert apply_gate({"ftsI": 0}, ["ftsI"], "susceptible") == "susceptible"


def test_silent_when_any_target_is_present():
    # At least one target present -> gate stays out of the way (returns None).
    assert apply_gate({"gyrA": 1, "parC": 0}, ["gyrA", "parC"], "no_call") is None
    assert apply_gate({"gyrA": 1, "parC": 1}, ["gyrA", "parC"], "no_call") is None


def test_unknown_target_does_not_fire():
    # A genome missing from the target table (target state unknown) must not be
    # forced — the model call stands and the gap is logged upstream.
    assert apply_gate({}, ["gyrA", "parC"], "no_call") is None
    assert apply_gate({"gyrA": 0}, ["gyrA", "parC"], "no_call") is None
