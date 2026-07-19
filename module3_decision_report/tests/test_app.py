"""Streamlit UI tests for the decision report rendering surface."""

from __future__ import annotations

import re

from streamlit.testing.v1 import AppTest

from decision_report.app import render_probability
from decision_report.config import DecisionConfig
from decision_report.contracts import DecisionLabel, DrugDecision, EvidenceCategory, NoCallReason
from decision_report.report import MANDATORY_DISCLAIMER


APP_PATH = "src/decision_report/app.py"


def _run_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=10).run()


def _all_text(at: AppTest) -> str:
    groups = (at.markdown, at.error, at.warning, at.info, at.success)
    return "\n".join(str(element.value) for group in groups for element in group)


def _select_case(at: AppTest, name: str) -> AppTest:
    at.selectbox[0].select(name)
    return at.run()


def _click(at: AppTest, label: str) -> AppTest:
    button = next(item for item in at.button if item.label == label)
    button.click()
    return at.run()


def _text_input(at: AppTest, label: str):
    return next(item for item in at.text_input if item.label == label)


def test_mock_mode_is_labelled_as_demonstration_data():
    """Fabricated output must never be mistakable for a clinical prediction."""
    text = _all_text(_run_app())

    assert "DEMONSTRATION DATA" in text
    assert "not a real prediction" in text


def test_demonstration_banner_survives_generating_a_report():
    """The banner qualifies the cards, so it must persist once they render."""
    at = _select_case(_run_app(), "Known mechanism (resistant)")
    at = _click(at, "Generate clinical report")

    assert "DEMONSTRATION DATA" in _all_text(at)


def test_fail_and_work_glyphs_differ_in_form_not_only_orientation():
    """An up/down triangle pair is the weakest possible non-colour distinction."""
    at = _select_case(_run_app(), "Known mechanism (resistant)")
    at = _click(at, "Generate clinical report")
    text = _all_text(at)

    assert "✓ Likely to work" in text
    assert "▲ Likely to fail" in text
    assert "▼" not in text


def test_disclaimer_is_present_on_first_paint_and_has_no_dismiss_control():
    at = _run_app()

    assert _all_text(at).count(MANDATORY_DISCLAIMER) >= 2
    dismiss_words = re.compile(r"dismiss|close|hide|don't show again", re.IGNORECASE)
    assert not any(dismiss_words.search(item.label) for item in [*at.button, *at.checkbox])


def test_disclaimer_remains_after_loading_a_report():
    at = _select_case(_run_app(), "Known mechanism (resistant)")
    at = _click(at, "Generate clinical report")

    assert MANDATORY_DISCLAIMER in _all_text(at)
    assert "Likely to fail" in _all_text(at)


def test_disclaimer_remains_on_unsupported_species_failure():
    at = _run_app()
    _text_input(at, "Report species").input("Klebsiella pneumoniae")
    at = at.run()
    at = _click(at, "Generate clinical report")

    assert MANDATORY_DISCLAIMER in _all_text(at)
    assert any("not covered by this pipeline" in str(item.value) for item in at.error)


def test_conflicting_evidence_routes_render_distinct_rationales():
    """The shared enum must not erase the clinically opposite conflict routes."""
    at_a = _select_case(_run_app(), "Conflict: mechanism vs model (no-call)")
    at_a = _click(at_a, "Generate clinical report")
    text_a = _all_text(at_a)

    at_b = _select_case(_run_app(), "Conflict: model vs no-signal (no-call)")
    at_b = _click(at_b, "Generate clinical report")
    text_b = _all_text(at_b)

    assert "A curated known resistance mechanism was detected" in text_a
    assert "no known resistance signal was found" in text_b
    assert text_a != text_b


def test_uncovered_drug_card_never_prints_none_or_nan():
    """Exercise all three nullable fields through a real uncovered-drug card."""
    at = _run_app()
    _text_input(at, "Additional requested drugs (comma-separated)").input("NovelDrug")
    at = at.run()
    at = _click(at, "Generate clinical report")
    text = _all_text(at)

    assert "No prediction available for this drug" in text
    assert "No clinical verdict was produced" in text
    assert not re.search(r"\b(?:None|nan)\b", text)


def test_intrinsic_card_omits_one_probability_panel():
    """The target-absence gate leaves four model-driven peers, but no fifth bar."""
    intrinsic = _select_case(_run_app(), "Intrinsic (no molecular target)")
    intrinsic = _click(intrinsic, "Generate clinical report")
    intrinsic_panels = sum(
        str(item.value).strip().startswith('<div class="probability-panel')
        for item in intrinsic.markdown
    )

    probabilistic = _select_case(_run_app(), "Known mechanism (resistant)")
    probabilistic = _click(probabilistic, "Generate clinical report")
    probabilistic_panels = sum(
        str(item.value).strip().startswith('<div class="probability-panel')
        for item in probabilistic.markdown
    )

    assert intrinsic_panels == 4
    assert probabilistic_panels == 5
    assert "Likely to fail — no molecular target (deterministic)" in _all_text(intrinsic)


def test_intrinsic_card_without_hits_names_absent_target_evidence():
    at = _select_case(_run_app(), "Intrinsic (no molecular target)")
    at = _click(at, "Generate clinical report")
    text = _all_text(at)

    assert "Known mechanism: absent molecular target (no gene hit required)" in text
    assert "(i) known mechanism" not in text


def test_out_of_range_probability_does_not_render_panel(monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr(
        "decision_report.app.st.markdown",
        lambda body, **kwargs: rendered.append(str(body)),
    )
    decision = DrugDecision(
        drug="RangeCheckDrug",
        label=DecisionLabel.NO_CALL,
        evidence_category=EvidenceCategory.NO_SIGNAL,
        calibrated_prob_resistant=1.7,
        target_present=True,
        intrinsic_resistance=False,
        no_call_reason=NoCallReason.INVALID_INPUT,
        ood_score=0.1,
        rationale="Invalid probability.",
    )

    render_probability(decision, DecisionConfig())

    markup = "\n".join(rendered)
    assert "probability-panel" not in markup
    assert "170.0%" not in markup
    assert "1.70" not in markup


def test_empty_features_show_caution_and_drug_cards():
    at = _select_case(_run_app(), "Susceptible (clean)")
    at = _click(at, "Generate clinical report")
    text = _all_text(at)

    assert "No features were extracted for this genome" in text
    assert "Likely to work" in text


def test_empty_fasta_extraction_failure_shows_error_without_drug_cards():
    at = _run_app()
    at.radio[0].set_value("Upload FASTA")
    at = at.run()
    at.file_uploader[0].set_value(("empty.fasta", b"", "text/plain"))
    at = at.run()
    at = _click(at, "Generate clinical report")
    text = _all_text(at)

    assert not at.exception
    assert "Feature extraction failed:" in text
    assert not any(
        str(item.value).strip().startswith('<div class="verdict-badge"')
        for item in at.markdown
    )


def test_evaluation_surfaces_no_call_rate_and_real_tables():
    at = _run_app()
    at = _click(at, "Run mock evaluation")

    assert "No-call rate" in _all_text(at)
    assert len(at.dataframe) >= 4
