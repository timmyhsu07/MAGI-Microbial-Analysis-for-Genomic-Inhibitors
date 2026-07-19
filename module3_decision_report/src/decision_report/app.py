"""Streamlit rendering surface for the GENOME FIREWALL decision report."""

from __future__ import annotations

import html
import math
import os
import tempfile
from dataclasses import asdict, replace
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from decision_report.config import DecisionConfig
from decision_report.contracts import (
    DecisionLabel,
    DecisionReportError,
    DrugDecision,
    EvidenceCategory,
    GenomeReport,
    NoCallReason,
)
from decision_report.decision import IMPORTANCE_CAVEAT
from decision_report.evaluation import run_evaluation
from decision_report.mock_pipeline import (
    MOCK_SPECIES,
    MockFeatureExtractor,
    MockPredictor,
    build_held_out_set,
    demo_cases,
    scripted_predictor_for,
)
from decision_report.real_pipeline import IntegrationError, ModelPredictor, Module1FeatureStore
from decision_report.report import MANDATORY_DISCLAIMER, build_report, report_from_fasta


EVIDENCE_LABELS = {
    EvidenceCategory.KNOWN_MECHANISM: "(i) known mechanism",
    EvidenceCategory.ASSOCIATION_ONLY: "(ii) statistical association only",
    EvidenceCategory.NO_SIGNAL: "(iii) no known resistance signal",
}

NO_CALL_LABELS = {
    NoCallReason.UNCERTAINTY_BAND: "probability inside the uncertainty band",
    NoCallReason.CONFLICTING_EVIDENCE: "conflicting evidence — see rationale below",
    NoCallReason.OUT_OF_DISTRIBUTION: "genome unlike training data (high OOD)",
    NoCallReason.DRUG_NOT_COVERED: "drug not covered by this predictor",
    NoCallReason.INVALID_INPUT: "invalid prediction input",
}


def inject_styles() -> None:
    """Apply the restrained, paper-like visual system once per app run."""
    st.markdown(
        """
        <style>
        :root {
            --paper: #faf9f7;
            --ink: #1a1a1a;
            --muted-ink: #514f4b;
            --rule: #cbc7c0;
            --fail: #8a1f11;
            --work: #1f6b3f;
            --nocall: #4a3d7a;
            --focus: #2456c9;
        }
        html, body, [class*="st-"] { font-size: 16px; }
        .stApp { background: var(--paper); color: var(--ink); }
        .stApp p, .stApp li, .stApp label { line-height: 1.5; }
        .stApp small { font-size: 14px; }
        :focus-visible { outline: 3px solid #2456c9; outline-offset: 2px; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        .disclaimer-banner {
            border: 2px solid #5c574f;
            background: #f2efe9;
            color: var(--ink);
            padding: 16px 18px;
            margin: 0 0 24px 0;
            line-height: 1.55;
            font-weight: 600;
            font-size: 16px;
        }
        .verdict-badge, .evidence-chip {
            display: inline-block;
            color: #ffffff;
            border-radius: 4px;
            font-weight: 700;
            line-height: 1.4;
        }
        .verdict-badge { padding: 9px 13px; margin: 2px 0 12px 0; font-size: 16px; }
        .evidence-chip { padding: 5px 9px; margin: 6px 0 12px 0; font-size: 14px; background: #55524e; }
        .probability-panel {
            border-left: 4px solid #77716a;
            padding: 10px 12px;
            margin: 8px 0 16px 0;
            background: #f4f1ec;
        }
        .probability-panel.muted { opacity: 0.78; }
        .prob-track { position: relative; height: 18px; background: #dedad3; margin: 10px 0 8px 0; }
        .prob-band { position: absolute; top: 0; bottom: 0; background: #b9b2cf; opacity: .8; }
        .prob-marker { position: absolute; top: -4px; width: 4px; height: 26px; background: #171717; }
        .mono, code, [data-testid="stDataFrame"] { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
        .feature-list { margin: 12px 0; }
        .feature-row {
            display: grid;
            grid-template-columns: minmax(120px, 1.1fr) 1fr 2px 1fr minmax(145px, 1.2fr);
            gap: 8px;
            align-items: center;
            margin: 10px 0;
            min-height: 26px;
        }
        .zero-line { width: 2px; height: 26px; background: #34312d; }
        .feature-bar { height: 14px; min-width: 3px; background: #4a3d7a; }
        .feature-left { margin-left: auto; }
        .feature-direction { color: #34312d; }
        .importance-note { border-left: 3px solid #857f77; padding: 8px 12px; margin: 12px 0; }
        .mode-banner {
            padding: 12px 16px; margin: 0 0 20px 0; font-size: 16px;
            line-height: 1.55; border-left: 6px solid;
        }
        .mode-mock { background: #f6efe0; border-left-color: #8a6116; color: #1a1a1a; }
        .mode-real { background: #e9eff6; border-left-color: #1f3d5c; color: #1a1a1a; }
        div[data-testid="stVerticalBlockBorderWrapper"] { margin-bottom: 24px; border-color: var(--rule); }
        div[data-testid="stExpander"] { margin-top: 8px; }
        @media (max-width: 700px) {
            .feature-row { grid-template-columns: 1fr; border-bottom: 1px solid var(--rule); padding-bottom: 10px; }
            .zero-line { display: none; }
            .feature-left { margin-left: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    """Render the mandatory, non-dismissible disclaimer at body-text size."""
    st.markdown(
        f'<div class="disclaimer-banner" role="note">{html.escape(MANDATORY_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_valid_probability(value: Any) -> bool:
    return _finite_number(value) and 0.0 <= value <= 1.0


def _safe_numeric(value: float | None, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if _finite_number(value) else "Not available"


def render_mode_banner(source: str) -> None:
    """State the provenance of the numbers below, permanently, in the report body.

    Mock output rendered without this is indistinguishable from a real clinical
    prediction to anyone who did not configure the session -- including anyone
    shown a screenshot. Sidebar text does not carry this; it has to sit with the
    cards it qualifies.
    """
    if source == "Mock (demo)":
        st.markdown(
            '<div class="mode-banner mode-mock" role="note">'
            "<strong>DEMONSTRATION DATA — not a real prediction.</strong> This session "
            "runs on the mock predictor, which fabricates plausible values in order to "
            "exercise every decision branch. Its drug set is illustrative and differs "
            "from the real trained pipeline's. Nothing below is a clinical result."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="mode-banner mode-real" role="note">'
            "<strong>REAL PIPELINE.</strong> Decisions below are computed from the "
            "Module 1 and Module 2 artifacts configured in the sidebar. Confirm every "
            "result by laboratory susceptibility testing before any clinical use."
            "</div>",
            unsafe_allow_html=True,
        )


def _verdict(decision: DrugDecision) -> tuple[str, str, str]:
    if decision.label is DecisionLabel.LIKELY_TO_FAIL and decision.intrinsic_resistance:
        return "■", "Likely to fail — no molecular target (deterministic)", "#8a1f11"
    if decision.label is DecisionLabel.LIKELY_TO_FAIL:
        return "▲", "Likely to fail", "#8a1f11"
    if decision.label is DecisionLabel.LIKELY_TO_WORK:
        # A check, not a down-triangle: an up/down triangle pair differs only by
        # orientation, and mistaking "likely to fail" for "likely to work" is the
        # worst error this UI can produce. All four glyphs differ in FORM.
        return "✓", "Likely to work", "#1f6b3f"
    reason = NO_CALL_LABELS.get(decision.no_call_reason, "reason unavailable — see rationale below")
    return "●", f"No-call — insufficient evidence: {reason}", "#4a3d7a"


def _probability_frame(probability: float, config: DecisionConfig) -> str:
    if config.uncertainty_band_low <= probability <= config.uncertainty_band_high:
        return "Within the uncertainty band; this estimate does not support a confident call."
    if probability > config.uncertainty_band_high:
        return "Above the resistance-side threshold."
    return "Below the susceptibility-side threshold."


def render_probability(decision: DrugDecision, config: DecisionConfig) -> None:
    """Render probability with threshold context, except for intrinsic calls."""
    if decision.intrinsic_resistance or not _is_valid_probability(decision.calibrated_prob_resistant):
        return
    probability = float(decision.calibrated_prob_resistant)
    low = config.uncertainty_band_low
    high = config.uncertainty_band_high
    muted = " muted" if decision.label is DecisionLabel.NO_CALL else ""
    label = "Model estimate (de-emphasized for no-call)" if muted else "Calibrated model estimate"
    st.markdown(
        f"""
        <div class="probability-panel{muted}">
          <strong>{label}:</strong> <span class="mono">{probability:.2f}</span>
          <div class="prob-track" role="img" aria-label="Resistance probability {probability:.2f}; uncertainty band {low:.2f} to {high:.2f}">
            <span class="prob-band" style="left:{low * 100:.1f}%;width:{(high - low) * 100:.1f}%"></span>
            <span class="prob-marker" style="left:calc({probability * 100:.1f}% - 2px)"></span>
          </div>
          <div>Uncertainty band: <span class="mono">{low:.2f}–{high:.2f}</span>. {_probability_frame(probability, config)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hits_frame(decision: DrugDecision) -> pd.DataFrame:
    rows = []
    for hit in decision.supporting_hits:
        rows.append(
            {
                "element_symbol": hit.element_symbol,
                "element_subtype": hit.element_subtype,
                "method": hit.method,
                "pct_identity": _safe_numeric(hit.pct_identity),
                "pct_coverage": _safe_numeric(hit.pct_coverage),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["element_symbol", "element_subtype", "method", "pct_identity", "pct_coverage"],
    )


def render_features(decision: DrugDecision) -> None:
    st.markdown("**Model feature contributions**")
    if not decision.top_features:
        st.write("No model feature contributions were supplied.")
    else:
        scale = max(abs(feature.contribution) for feature in decision.top_features) or 1.0
        rows = []
        for feature in decision.top_features:
            width = max(2.0, 100.0 * abs(feature.contribution) / scale)
            known = "known mechanism" if feature.is_known_mechanism else "statistical only"
            name = html.escape(feature.name)
            value = feature.contribution
            if value >= 0:
                left_bar = ""
                right_bar = f'<div class="feature-bar" style="width:{width:.1f}%"></div>'
                direction = "▶ toward resistant"
            else:
                left_bar = f'<div class="feature-bar feature-left" style="width:{width:.1f}%"></div>'
                right_bar = ""
                direction = "◀ toward susceptible"
            rows.append(
                "<div class=\"feature-row\">"
                f"<div><code>{name}</code><br>{known}</div>"
                f"<div>{left_bar}</div><div class=\"zero-line\"></div><div>{right_bar}</div>"
                f"<div class=\"feature-direction\">{direction}<br><span class=\"mono\">{value:+.3f}</span></div>"
                "</div>"
            )
        st.markdown(f'<div class="feature-list">{"".join(rows)}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="importance-note">{html.escape(IMPORTANCE_CAVEAT)}</div>',
        unsafe_allow_html=True,
    )


def render_evidence_detail(decision: DrugDecision, config: DecisionConfig) -> None:
    with st.expander("Evidence detail"):
        st.markdown("**Supporting hits**")
        if decision.supporting_hits:
            st.dataframe(_hits_frame(decision), width="stretch", hide_index=True)
        else:
            st.write("No supporting hits.")
        render_features(decision)
        if _finite_number(decision.ood_score):
            score = float(decision.ood_score)
            comparison = "at or above" if score >= config.ood_threshold else "below"
            st.markdown(
                "OOD score: "
                f'<span class="mono">{score:.2f}</span> ({comparison} threshold: '
                f'<span class="mono">{config.ood_threshold:.2f}</span>).',
                unsafe_allow_html=True,
            )
        else:
            st.write("OOD score was not supplied for this decision.")


def render_drug_card(decision: DrugDecision, config: DecisionConfig) -> None:
    """Render one full-size card without reinterpreting the decision."""
    with st.container(border=True):
        st.markdown(f"### {html.escape(decision.drug)}")
        shape, verdict_text, color = _verdict(decision)
        st.markdown(
            f'<div class="verdict-badge" style="background:{color}">{shape} {html.escape(verdict_text)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(html.escape(decision.rationale))
        if decision.intrinsic_resistance and not decision.supporting_hits:
            tier = "Known mechanism: absent molecular target (no gene hit required)"
        else:
            tier = EVIDENCE_LABELS[decision.evidence_category]
        st.markdown(f'<span class="evidence-chip">Evidence tier: {html.escape(tier)}</span>', unsafe_allow_html=True)

        if decision.evidence_category is EvidenceCategory.NO_SIGNAL:
            st.write("No resistance determinants detected for this drug class.")

        render_probability(decision, config)

        if decision.no_call_reason is NoCallReason.DRUG_NOT_COVERED:
            st.markdown("**No prediction available for this drug. No clinical verdict was produced.**")
        else:
            render_evidence_detail(decision, config)

        if decision.caveats:
            st.markdown("**Caveats**")
            for caveat in decision.caveats:
                st.markdown(f"- {html.escape(caveat)}")


def render_coverage(report: GenomeReport) -> None:
    st.markdown("**Coverage recap**")
    covered = ", ".join(report.covered_drugs) if report.covered_drugs else "No covered drugs reported"
    st.write(f"Covered by the active predictor: {covered}.")
    if report.uncovered_drugs_requested:
        st.write("Requested but not covered: " + ", ".join(report.uncovered_drugs_requested) + ".")
    else:
        st.write("No requested drugs fell outside predictor coverage.")


def render_report(report: GenomeReport, config: DecisionConfig) -> None:
    if not report.species_supported:
        for message in report.errors:
            st.error(message)
        supported = ", ".join(config.covered_species) or "No species reported"
        st.write(f"Supported species: {supported}.")
        return

    if not report.decisions and report.errors:
        for message in report.errors:
            st.error(message)
        return

    if report.errors and report.decisions:
        for message in report.errors:
            st.warning(message)

    st.markdown(
        f"**Genome:** `{html.escape(report.genome_id)}`  \n**Species:** {html.escape(report.species)}"
    )
    for decision in report.decisions:
        render_drug_card(decision, config)
    render_coverage(report)


def _text_from_sequence(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(value.strip() for value in values if value.strip())


def _sidebar() -> tuple[str, Any, Any, DecisionConfig, str]:
    """Render controls and return source, predictor, extractor, config, species."""
    st.sidebar.markdown("**Data source**")
    source = st.sidebar.radio("Pipeline", ["Mock (demo)", "Real pipeline"], index=0)
    previous_source = st.session_state.get("active_source")
    if previous_source is not None and previous_source != source:
        st.session_state.pop("clinical_report", None)
        st.session_state.pop("clinical_config", None)
    st.session_state["active_source"] = source
    base_config = DecisionConfig()
    predictor: Any = None
    extractor: Any = None
    species = MOCK_SPECIES

    if source == "Mock (demo)":
        predictor = MockPredictor()
        extractor = MockFeatureExtractor()
    else:
        st.sidebar.write("Real mode reads existing Module 1 and Module 2 artifacts.")
        models_dir = st.sidebar.text_input("Models directory")
        target_gene_table = st.sidebar.text_input("Target gene table (CSV path)")
        module1_output_dir = st.sidebar.text_input("Module 1 output directory")
        species = st.sidebar.text_input("Species", value=MockPredictor().covered_species()[0])
        if models_dir and target_gene_table and module1_output_dir and species:
            try:
                predictor = ModelPredictor(models_dir, target_gene_table, species)
                extractor = Module1FeatureStore(module1_output_dir)
            except (IntegrationError, DecisionReportError, OSError, ValueError) as exc:
                st.sidebar.error(f"Real pipeline unavailable: {exc}")
            except Exception as exc:  # defensive boundary: never expose Streamlit traceback
                st.sidebar.error(f"Real pipeline unavailable: {exc}")
        else:
            st.sidebar.info("Enter all four real-pipeline fields to inspect live coverage.")

    covered_species = predictor.covered_species() if predictor is not None else []
    covered_drugs = predictor.covered_drugs() if predictor is not None else []
    requested = st.sidebar.text_input(
        "Additional requested drugs (comma-separated)",
        help="Optional. Uncovered names render as first-class no-call cards.",
    )
    additions = _text_from_sequence(requested.split(","))
    drugs_of_interest = tuple(dict.fromkeys([*covered_drugs, *additions])) if additions else ()
    config = replace(
        base_config,
        covered_species=tuple(covered_species) if covered_species else base_config.covered_species,
        drugs_of_interest=drugs_of_interest,
        ood_threshold=(predictor.ood_threshold() if source == "Real pipeline" and predictor is not None else base_config.ood_threshold),
    )

    st.sidebar.markdown("**Live predictor coverage**")
    st.sidebar.write("Species: " + (", ".join(covered_species) if covered_species else "Unavailable"))
    st.sidebar.write("Drugs: " + (", ".join(covered_drugs) if covered_drugs else "Unavailable"))
    st.sidebar.markdown("**Decision configuration provenance**")
    st.sidebar.write(config.tuned_on)
    st.sidebar.markdown(
        "Uncertainty band: "
        f'<span class="mono">[{config.uncertainty_band_low:.2f}, {config.uncertainty_band_high:.2f}]</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'OOD threshold: <span class="mono">{config.ood_threshold:.2f}</span>',
        unsafe_allow_html=True,
    )
    return source, predictor, extractor, config, species


def _run_uploaded_mock(upload: Any, predictor: Any, extractor: Any, config: DecisionConfig, species: str) -> GenomeReport:
    suffix = os.path.splitext(upload.name)[1] or ".fasta"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(upload.getvalue())
            temp_path = handle.name
        return report_from_fasta(temp_path, extractor, predictor, config, species)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def render_clinical_tab(source: str, predictor: Any, extractor: Any, config: DecisionConfig, species: str) -> None:
    st.markdown("## Clinical Report")
    render_mode_banner(source)
    report: GenomeReport | None = st.session_state.get("clinical_report")
    report_config: DecisionConfig = st.session_state.get("clinical_config", config)

    if source == "Mock (demo)":
        input_mode = st.radio("Input method", ["Demo case", "Upload FASTA"], horizontal=True)
        cases = demo_cases()
        selected_name = st.selectbox("Demo case", [case.name for case in cases])
        selected_case = next(case for case in cases if case.name == selected_name)
        st.write(selected_case.description)
        selected_species = st.text_input("Report species", value=selected_case.species)
        upload = None
        if input_mode == "Upload FASTA":
            upload = st.file_uploader("FASTA file", type=["fasta", "fa", "fna", "fas"])

        if st.button("Generate clinical report", type="primary"):
            try:
                with st.spinner("Running the decision pipeline over this genome..."):
                    if input_mode == "Demo case":
                        scripted = scripted_predictor_for(cases)
                        report = build_report(selected_case.features, scripted, config, selected_species)
                    elif upload is not None:
                        report = _run_uploaded_mock(upload, predictor, extractor, config, selected_species)
                    else:
                        report = None
                        st.session_state.pop("clinical_report", None)
                        st.error("Choose a FASTA file before generating a report.")
                st.session_state["clinical_report"] = report
                st.session_state["clinical_config"] = config
            except (DecisionReportError, IntegrationError) as exc:
                report = None
                st.session_state.pop("clinical_report", None)
                st.error(f"Decision pipeline unavailable: {exc}")
            except Exception as exc:
                report = None
                st.session_state.pop("clinical_report", None)
                st.error(f"The decision pipeline could not complete: {exc}")
    else:
        st.write("Enter an identifier already present in the configured Module 1 feature store.")
        identifier = st.text_input("Genome ID or original FASTA filename")
        if st.button("Generate clinical report", type="primary"):
            if predictor is None or extractor is None:
                report = None
                st.session_state.pop("clinical_report", None)
                st.error("Real pipeline is not available. Check the sidebar artifact paths.")
            elif not identifier.strip():
                report = None
                st.session_state.pop("clinical_report", None)
                st.error("Enter a genome ID or FASTA filename before generating a report.")
            else:
                try:
                    with st.spinner("Running the decision pipeline over this genome..."):
                        features = extractor(identifier.strip())
                        report = build_report(features, predictor, config, species)
                    st.session_state["clinical_report"] = report
                    st.session_state["clinical_config"] = config
                except (DecisionReportError, IntegrationError) as exc:
                    report = None
                    st.session_state.pop("clinical_report", None)
                    st.error(f"Decision pipeline unavailable: {exc}")
                except Exception as exc:
                    report = None
                    st.session_state.pop("clinical_report", None)
                    st.error(f"The decision pipeline could not complete: {exc}")

    if report is None:
        st.info("Choose a demo case or provide an input, then generate a report. No-call outcomes are expected when evidence is insufficient.")
        if predictor is not None:
            st.write("Covered species: " + ", ".join(predictor.covered_species()) + ".")
            st.write("Covered drugs: " + ", ".join(predictor.covered_drugs()) + ".")
    else:
        render_report(report, report_config)

    render_disclaimer()


def _display_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "Not available"
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def render_evaluation_tab() -> None:
    st.markdown("## Evaluation")
    st.write("Mock held-out evaluation only. These aggregate metrics are kept separate from an individual genome report.")
    n = int(st.number_input("Held-out genomes", min_value=10, max_value=500, value=60, step=10))
    seed = int(st.number_input("Random seed", min_value=0, value=20260718, step=1))
    if st.button("Run mock evaluation"):
        try:
            with st.spinner("Running held-out evaluation..."):
                eval_predictor = MockPredictor()
                eval_config = replace(DecisionConfig(), covered_species=tuple(eval_predictor.covered_species()))
                held_out = build_held_out_set(n=n, seed=seed)
                st.session_state["evaluation_result"] = run_evaluation(
                    held_out, eval_predictor, eval_config, species=MOCK_SPECIES
                )
        except Exception as exc:
            st.error(f"Evaluation could not complete: {exc}")

    result = st.session_state.get("evaluation_result")
    if result is None:
        st.info("Set the held-out sample size and seed, then run the mock evaluation.")
        return

    overall_frame = pd.DataFrame(
        [{"metric": key, "value": _display_value(value)} for key, value in result.overall.items()]
    )
    st.markdown("**Overall performance**")
    st.dataframe(overall_frame, width="stretch", hide_index=True)
    no_call = result.overall.get("no_call_rate")
    if _finite_number(no_call):
        st.markdown(
            f'No-call rate: <span class="mono">{float(no_call):.3f}</span>. '
            "This is reported explicitly rather than treating abstentions as confident predictions.",
            unsafe_allow_html=True,
        )

    st.markdown("**Per-drug performance**")
    st.dataframe(result.per_drug, width="stretch", hide_index=True)
    st.markdown("**Performance by genetic group**")
    st.dataframe(result.per_group, width="stretch", hide_index=True)
    st.markdown("**Reliability by probability bin**")
    reliability = pd.DataFrame([asdict(item) for item in result.reliability])
    if reliability.empty:
        st.write("No reliability bins were available.")
    else:
        st.dataframe(reliability, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="GENOME FIREWALL", page_icon="🧬", layout="wide")
    inject_styles()
    st.title("GENOME FIREWALL")
    render_disclaimer()
    source, predictor, extractor, config, species = _sidebar()
    clinical_tab, evaluation_tab = st.tabs(["Clinical Report", "Evaluation"])
    with clinical_tab:
        render_clinical_tab(source, predictor, extractor, config, species)
    with evaluation_tab:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
