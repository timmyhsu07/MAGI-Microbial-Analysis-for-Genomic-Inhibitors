"""Streamlit rendering surface for the MAGI decision report."""

from __future__ import annotations

import html
import math
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

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

VIEW_ANALYZE = "analyze"
VIEW_REPORT = "report"
VIEW_EVALUATION = "evaluation"
PRODUCT_NAME = "MAGI"
PRODUCT_LONG_NAME = "Microbial Analysis for Genomic Inhibitors"


def inject_styles() -> None:
    """Apply the responsive pastel-green visual system once per app run."""
    st.markdown(
        """
        <style>
        :root {
            --canvas: #f4faf5;
            --surface: #ffffff;
            --surface-soft: #edf7ef;
            --surface-strong: #dff0e3;
            --mint: #cce8d3;
            --sage: #7fa48a;
            --forest: #183e2b;
            --forest-soft: #2f674a;
            --ink: #173126;
            --muted-ink: #587064;
            --rule: #cee2d3;
            --rule-strong: #a9c9b2;
            --fail: #a4493d;
            --work: #2f7553;
            --nocall: #6c5b89;
            --focus: #267a54;
            --shadow-sm: 0 8px 24px rgba(40, 91, 62, 0.07);
            --shadow-md: 0 18px 48px rgba(40, 91, 62, 0.10);
        }
        /* Let typography inherit so Streamlit's icon classes keep their
           Material Symbols font instead of exposing names like arrow_right. */
        html, body, .stApp {
            font-family: "Avenir Next", "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 16px;
        }
        html {
            color-scheme: light;
            scroll-behavior: smooth;
        }
        .stApp {
            background:
                radial-gradient(circle at 8% 4%, rgba(188, 229, 198, 0.44), transparent 29rem),
                radial-gradient(circle at 92% 10%, rgba(223, 240, 227, 0.75), transparent 24rem),
                var(--canvas);
            color: var(--ink);
        }
        .stApp p, .stApp li, .stApp label { color: var(--ink); line-height: 1.6; }
        .stApp small, .stCaption { color: var(--muted-ink); font-size: 0.875rem; }
        .stApp h1, .stApp h2, .stApp h3 {
            color: var(--forest);
            letter-spacing: -0.025em;
        }
        :focus-visible {
            outline: 3px solid var(--focus) !important;
            outline-offset: 3px;
            border-radius: 8px;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { right: 1rem; }
        .block-container {
            width: min(100%, 1240px);
            padding: 1rem clamp(1rem, 3vw, 2.5rem) 3rem;
        }
        .hero-shell {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--rule);
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(255,255,255,.96), rgba(225,243,229,.96));
            box-shadow: var(--shadow-md);
            padding: clamp(1.35rem, 3vw, 2.35rem);
            margin-bottom: 1rem;
        }
        .hero-shell::after {
            content: "";
            position: absolute;
            width: 220px;
            height: 220px;
            right: -65px;
            top: -80px;
            border: 38px solid rgba(105, 158, 119, .12);
            border-radius: 50%;
        }
        .hero-eyebrow, .section-kicker, .sidebar-eyebrow {
            color: var(--forest-soft);
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .14em;
            text-transform: uppercase;
        }
        .hero-title {
            max-width: 820px;
            color: var(--forest);
            font-size: clamp(2.2rem, 5vw, 3.8rem);
            line-height: 1;
            letter-spacing: -.055em;
            margin: .45rem 0 .7rem;
        }
        .hero-copy {
            max-width: 720px;
            color: var(--muted-ink);
            font-size: clamp(1rem, 2vw, 1.16rem);
            line-height: 1.65;
            margin: 0;
        }
        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            gap: .55rem;
            margin-top: 1rem;
        }
        .hero-pill {
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            border: 1px solid var(--rule-strong);
            border-radius: 999px;
            background: rgba(255,255,255,.72);
            color: var(--forest);
            font-size: .82rem;
            font-weight: 700;
            padding: .48rem .75rem;
        }
        .hero-dot {
            width: .52rem;
            height: .52rem;
            border-radius: 50%;
            background: #5e9a70;
            box-shadow: 0 0 0 4px rgba(94,154,112,.14);
        }
        .section-heading {
            color: var(--forest);
            font-size: clamp(1.65rem, 3vw, 2.35rem);
            line-height: 1.15;
            margin: .3rem 0 .65rem;
        }
        .page-masthead {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 1rem;
            border-bottom: 1px solid var(--rule);
            padding: .35rem 0 1rem;
            margin-bottom: 1rem;
        }
        .page-masthead h1 {
            font-size: clamp(1.8rem, 4vw, 2.7rem);
            line-height: 1.05;
            margin: .2rem 0 0;
        }
        .page-masthead p {
            max-width: 520px;
            color: var(--muted-ink);
            font-size: .9rem;
            margin: 0;
        }
        .disclaimer-banner {
            display: flex;
            align-items: flex-start;
            gap: .8rem;
            border: 1px solid #b9d5bf;
            border-radius: 16px;
            background: rgba(234, 246, 236, .94);
            color: var(--ink);
            box-shadow: var(--shadow-sm);
            padding: .9rem 1rem;
            margin: 0 0 1.3rem;
            line-height: 1.55;
            font-weight: 650;
            font-size: .92rem;
        }
        .disclaimer-icon {
            display: grid;
            place-items: center;
            flex: 0 0 2rem;
            width: 2rem;
            height: 2rem;
            border-radius: 10px;
            background: var(--mint);
            color: var(--forest);
            font-size: 1rem;
        }
        .verdict-badge, .evidence-chip {
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            color: #ffffff;
            border-radius: 999px;
            font-weight: 700;
            line-height: 1.4;
        }
        .verdict-badge {
            padding: .55rem .82rem;
            margin: .1rem 0 .85rem;
            font-size: .88rem;
            box-shadow: 0 5px 14px rgba(25,62,43,.12);
        }
        .evidence-chip {
            padding: .42rem .7rem;
            margin: .35rem 0 .85rem;
            font-size: .78rem;
            background: var(--forest-soft);
        }
        .drug-heading {
            font-size: 1.3rem;
            margin: 0 0 .55rem;
        }
        .probability-panel {
            border: 1px solid var(--rule);
            border-radius: 14px;
            padding: .85rem;
            margin: .5rem 0 1rem;
            background: var(--surface-soft);
            font-size: .9rem;
        }
        .probability-panel.muted { background: #f2f0f7; border-color: #d9d2e5; }
        .prob-track {
            position: relative;
            height: 12px;
            overflow: visible;
            border-radius: 999px;
            background: #cfe3d4;
            margin: .85rem 0 .7rem;
        }
        .prob-band {
            position: absolute;
            top: 0;
            bottom: 0;
            background: #b9aece;
            opacity: .86;
        }
        .prob-marker {
            position: absolute;
            top: -4px;
            width: 4px;
            height: 20px;
            border-radius: 99px;
            background: var(--forest);
            box-shadow: 0 0 0 3px rgba(255,255,255,.8);
        }
        .mono, code, [data-testid="stDataFrame"] { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
        code {
            border-radius: 6px;
            background: var(--surface-strong);
            color: var(--forest);
            padding: .12rem .3rem;
        }
        .feature-list { margin: .75rem 0; }
        .feature-row {
            display: grid;
            grid-template-columns: minmax(120px, 1.1fr) 1fr 2px 1fr minmax(145px, 1.2fr);
            gap: .55rem;
            align-items: center;
            border-bottom: 1px solid var(--rule);
            padding: .6rem 0;
            min-height: 2rem;
            font-size: .85rem;
        }
        .feature-row:last-child { border-bottom: 0; }
        .zero-line { width: 2px; height: 1.65rem; background: var(--sage); }
        .feature-bar { height: .75rem; min-width: 3px; border-radius: 99px; background: var(--forest-soft); }
        .feature-left { margin-left: auto; }
        .feature-direction { color: var(--muted-ink); }
        .importance-note {
            border: 1px solid var(--rule);
            border-radius: 12px;
            background: var(--surface-soft);
            color: var(--muted-ink);
            padding: .75rem .85rem;
            margin: .8rem 0;
            font-size: .86rem;
        }
        .mode-banner {
            border: 1px solid;
            border-radius: 16px;
            padding: .9rem 1rem;
            margin: 0 0 1.1rem;
            font-size: .91rem;
            line-height: 1.55;
        }
        .mode-mock { background: #fff8e9; border-color: #ead6a6; color: #5e4a1f; }
        .mode-real { background: #e8f5eb; border-color: #add1b7; color: var(--forest); }
        .report-identity, .coverage-panel {
            border: 1px solid var(--rule);
            border-radius: 16px;
            background: rgba(255,255,255,.76);
            box-shadow: var(--shadow-sm);
            padding: 1rem 1.1rem;
            margin: .65rem 0 1rem;
        }
        .report-identity {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem 2.5rem;
        }
        .meta-label {
            display: block;
            color: var(--muted-ink);
            font-size: .7rem;
            font-weight: 800;
            letter-spacing: .11em;
            text-transform: uppercase;
            margin-bottom: .2rem;
        }
        .meta-value { color: var(--forest); font-size: .95rem; font-weight: 700; }
        .coverage-panel strong { color: var(--forest); }

        /* Streamlit primitives */
        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--rule);
            background: rgba(235, 247, 238, .96);
        }
        section[data-testid="stSidebar"] > div { padding-top: .8rem; }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .55rem; }
        section[data-testid="stSidebar"] hr { margin: .35rem 0; }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted-ink); }
        .sidebar-brand {
            color: var(--forest);
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: -.02em;
            margin: .15rem 0 .65rem;
        }
        .sidebar-panel {
            border: 1px solid var(--rule);
            border-radius: 14px;
            background: rgba(255,255,255,.72);
            padding: .75rem .8rem;
            margin: .2rem 0;
        }
        .sidebar-panel + .sidebar-panel { margin-top: .55rem; }
        .sidebar-value {
            display: block;
            color: var(--muted-ink);
            font-size: .84rem;
            line-height: 1.45;
            margin-top: .18rem;
        }
        main div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--rule);
            border-radius: 20px;
            background: rgba(255,255,255,.88);
            box-shadow: var(--shadow-sm);
            margin-bottom: 1rem;
            overflow: hidden;
        }
        div[data-testid="stExpander"] {
            border-color: var(--rule);
            border-radius: 12px;
            background: var(--surface);
            margin-top: .5rem;
            overflow: hidden;
        }
        div[data-testid="stAlert"] { border-radius: 14px; }
        [data-testid="stDataFrame"] {
            max-width: 100%;
            overflow-x: auto;
            border: 1px solid var(--rule);
            border-radius: 14px;
        }
        .stButton > button {
            min-height: 2.9rem;
            border-radius: 12px;
            border-color: var(--rule-strong);
            background: rgba(255,255,255,.92);
            color: var(--forest) !important;
            font-weight: 750;
            transition: transform .16s ease, box-shadow .16s ease, background .16s ease;
        }
        .stButton > button p,
        .stButton > button span {
            color: inherit !important;
        }
        .stButton > button:hover {
            border-color: var(--forest-soft);
            background: var(--surface-strong);
            color: var(--forest);
            transform: translateY(-1px);
            box-shadow: var(--shadow-sm);
        }
        .stButton > button[kind="primary"] {
            border-color: var(--forest-soft);
            background: var(--forest-soft);
            color: #ffffff;
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--forest);
            color: #ffffff;
        }
        .stButton > button:disabled {
            border-color: var(--rule);
            background: #e7f0e9;
            color: #718278 !important;
            opacity: 1;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        [data-testid="stFileUploaderDropzone"] {
            border-color: var(--rule-strong) !important;
            border-radius: 12px !important;
            background: rgba(255,255,255,.88) !important;
            color: var(--ink) !important;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] [role="combobox"] {
            background: transparent !important;
            color: var(--ink) !important;
            caret-color: var(--forest);
            -webkit-text-fill-color: var(--ink) !important;
        }
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stNumberInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder {
            color: #789083 !important;
            opacity: 1;
            -webkit-text-fill-color: #789083 !important;
        }
        [data-baseweb="select"] svg {
            fill: var(--forest) !important;
        }
        [data-baseweb="popover"],
        [role="listbox"],
        [role="option"] {
            background: var(--surface) !important;
            color: var(--ink) !important;
        }
        [role="option"]:hover,
        [role="option"][aria-selected="true"] {
            background: var(--surface-strong) !important;
            color: var(--forest) !important;
        }
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] label p {
            color: var(--ink) !important;
        }
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploaderDropzone"] button {
            color: var(--forest) !important;
        }
        div[role="radiogroup"] { gap: .45rem; }
        div[role="radiogroup"] label {
            border-radius: 10px;
            padding: .35rem .5rem;
        }

        @media (max-width: 900px) {
            .block-container { padding-top: .7rem; }
            .hero-shell { border-radius: 22px; }
            [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
            [data-testid="column"] {
                flex: 1 1 min(100%, 25rem) !important;
                width: min(100%, 25rem) !important;
                min-width: 0 !important;
            }
        }
        @media (max-width: 640px) {
            html, body, .stApp { font-size: 15px; }
            .block-container { padding: .75rem .8rem 2.5rem; }
            .hero-shell { padding: 1.15rem 1rem; margin-bottom: .8rem; }
            .hero-title { font-size: clamp(2.15rem, 13vw, 3rem); }
            .hero-copy { font-size: .95rem; }
            .hero-pill { font-size: .75rem; }
            .page-masthead { display: block; }
            .page-masthead p { margin-top: .55rem; }
            .disclaimer-banner { padding: .8rem; font-size: .84rem; }
            .disclaimer-icon { flex-basis: 1.75rem; width: 1.75rem; height: 1.75rem; }
            .mode-banner { font-size: .86rem; }
            .report-identity { display: grid; gap: .75rem; }
            .feature-row { grid-template-columns: 1fr; gap: .25rem; }
            .zero-line { display: none; }
            .feature-row > div:nth-child(2),
            .feature-row > div:nth-child(4) { display: none; }
            .feature-left { margin-left: 0; }
            .stButton > button { width: 100%; }
            button[data-baseweb="tab"] { padding-inline: .65rem; font-size: .87rem; }
            [data-testid="stFileUploaderDropzone"] { padding: .75rem !important; }
        }
        @media (prefers-reduced-motion: reduce) {
            html { scroll-behavior: auto; }
            *, *::before, *::after {
                animation-duration: .01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: .01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    """Render the mandatory, non-dismissible disclaimer at body-text size."""
    st.markdown(
        '<div class="disclaimer-banner" role="note">'
        '<span class="disclaimer-icon" aria-hidden="true">✦</span>'
        f"<span>{html.escape(MANDATORY_DISCLAIMER)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the product identity and scope without relying on Streamlit's title."""
    st.markdown(
        f"""
        <header class="hero-shell">
          <div class="hero-eyebrow">{html.escape(PRODUCT_LONG_NAME)}</div>
          <h1 class="hero-title">{html.escape(PRODUCT_NAME)}</h1>
          <p class="hero-copy">
            Evidence-aware antibiotic resistance screening from assembled bacterial genomes.
            Every result separates known mechanisms, statistical signals, and honest no-calls.
          </p>
          <div class="hero-pills" aria-label="Pipeline qualities">
            <span class="hero-pill"><span class="hero-dot"></span>Read-only analysis</span>
            <span class="hero-pill"><span class="hero-dot"></span>Auditable evidence</span>
            <span class="hero-pill"><span class="hero-dot"></span>Laboratory confirmation required</span>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_page_masthead(title: str, eyebrow: str, description: str) -> None:
    """Render a compact header for focused secondary views."""
    st.markdown(
        '<header class="page-masthead">'
        f'<div><div class="section-kicker">{html.escape(eyebrow)}</div>'
        f"<h1>{html.escape(title)}</h1></div>"
        f"<p>{html.escape(description)}</p>"
        "</header>",
        unsafe_allow_html=True,
    )


def _open_view(view: str) -> None:
    st.session_state["active_view"] = view
    st.rerun()


def render_navigation(active_view: str) -> None:
    """Render page-level navigation without keeping every page mounted."""
    has_report = st.session_state.get("clinical_report") is not None
    labels = [
        ("Analyze", VIEW_ANALYZE),
        ("Evaluation", VIEW_EVALUATION),
    ]
    if has_report:
        labels.append(("Clinical report", VIEW_REPORT))

    columns = st.columns(len(labels), gap="small")
    for column, (label, view) in zip(columns, labels):
        with column:
            if st.button(
                label,
                key=f"navigate-{view}",
                type="primary" if active_view == view else "secondary",
                use_container_width=True,
            ):
                _open_view(view)


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
        return "■", "Likely to fail — no molecular target (deterministic)", "var(--fail)"
    if decision.label is DecisionLabel.LIKELY_TO_FAIL:
        return "▲", "Likely to fail", "var(--fail)"
    if decision.label is DecisionLabel.LIKELY_TO_WORK:
        # A check, not a down-triangle: an up/down triangle pair differs only by
        # orientation, and mistaking "likely to fail" for "likely to work" is the
        # worst error this UI can produce. All four glyphs differ in FORM.
        return "✓", "Likely to work", "var(--work)"
    reason = NO_CALL_LABELS.get(decision.no_call_reason, "reason unavailable — see rationale below")
    return "●", f"No-call — insufficient evidence: {reason}", "var(--nocall)"


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
            st.dataframe(_hits_frame(decision), use_container_width=True, hide_index=True)
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
        st.markdown(
            f'<h3 class="drug-heading">{html.escape(decision.drug)}</h3>',
            unsafe_allow_html=True,
        )
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
    covered = ", ".join(report.covered_drugs) if report.covered_drugs else "No covered drugs reported"
    uncovered = (
        "Requested but not covered: " + ", ".join(report.uncovered_drugs_requested) + "."
        if report.uncovered_drugs_requested
        else "No requested drugs fell outside predictor coverage."
    )
    st.markdown(
        '<div class="coverage-panel">'
        "<strong>Coverage recap</strong><br>"
        f"Covered by the active predictor: {html.escape(covered)}.<br>"
        f"{html.escape(uncovered)}"
        "</div>",
        unsafe_allow_html=True,
    )


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
        '<div class="report-identity">'
        '<div><span class="meta-label">Genome</span>'
        f'<span class="meta-value mono">{html.escape(report.genome_id)}</span></div>'
        '<div><span class="meta-label">Species</span>'
        f'<span class="meta-value">{html.escape(report.species)}</span></div>'
        '<div><span class="meta-label">Decision cards</span>'
        f'<span class="meta-value">{len(report.decisions)}</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    for start in range(0, len(report.decisions), 2):
        row = report.decisions[start : start + 2]
        columns = st.columns(2, gap="large")
        for column, decision in zip(columns, row):
            with column:
                render_drug_card(decision, config)
    render_coverage(report)


# Repo-relative location of the demo artifacts produced by
# scripts/fetch_bvbrc_ecoli.py + Module 1 + Module 2 (see README). Prefilling
# these means real mode loads with zero typing on stage; each is overridable via
# an env var for a non-default deployment.
_DEFAULT_DATA_SUBDIR = "data/bvbrc_ecoli"


def _demo_defaults() -> dict[str, str]:
    data = Path(__file__).resolve().parents[3] / _DEFAULT_DATA_SUBDIR
    return {
        "models_dir": os.environ.get("MAGI_MODELS_DIR", str(data / "module2_out" / "models")),
        "target_gene_table": os.environ.get("MAGI_TARGET_GENES", str(data / "target_genes.csv")),
        "module1_output_dir": os.environ.get("MAGI_MODULE1_OUT", str(data / "module1_out")),
        "species": os.environ.get("MAGI_SPECIES", "Escherichia coli"),
    }


def _sidebar() -> tuple[str, Any, Any, DecisionConfig, str]:
    """Render the compact analysis controls and resolve the active pipeline."""
    st.sidebar.markdown(
        '<div class="sidebar-eyebrow">Analysis workspace</div>'
        f'<div class="sidebar-brand">{html.escape(PRODUCT_NAME)}</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("**Data source**")
    source = st.sidebar.radio("Pipeline", ["Mock (demo)", "Real pipeline"], index=0)
    previous_source = st.session_state.get("active_source")
    if previous_source is not None and previous_source != source:
        st.session_state.pop("clinical_report", None)
        st.session_state.pop("clinical_config", None)
        st.session_state["active_view"] = VIEW_ANALYZE
    st.session_state["active_source"] = source
    base_config = DecisionConfig()
    predictor: Any = None
    extractor: Any = None
    species = MOCK_SPECIES

    if source == "Mock (demo)":
        predictor = MockPredictor()
        extractor = MockFeatureExtractor()
    else:
        defaults = _demo_defaults()
        demo_ready = (
            Path(defaults["module1_output_dir"]).exists()
            and Path(defaults["target_gene_table"]).exists()
            and Path(defaults["models_dir"]).exists()
        )
        if demo_ready:
            st.sidebar.caption("Bundled artifacts detected and ready.")
        else:
            st.sidebar.caption("Bundled artifacts were not found. Configure paths below.")
        with st.sidebar.expander("Artifact paths", expanded=not demo_ready):
            models_dir = st.text_input("Models directory", value=defaults["models_dir"])
            target_gene_table = st.text_input("Target gene table (CSV path)", value=defaults["target_gene_table"])
            module1_output_dir = st.text_input("Module 1 output directory", value=defaults["module1_output_dir"])
            species = st.text_input("Species", value=defaults["species"])
        if models_dir and target_gene_table and module1_output_dir and species:
            try:
                predictor = ModelPredictor(models_dir, target_gene_table, species)
                extractor = Module1FeatureStore(module1_output_dir)
            except Exception as exc:  # defensive boundary: never expose Streamlit traceback
                st.sidebar.error(f"Real pipeline unavailable: {exc}")
        else:
            st.sidebar.info("Enter all four real-pipeline fields to inspect live coverage.")

    covered_species = predictor.covered_species() if predictor is not None else []
    covered_drugs = predictor.covered_drugs() if predictor is not None else []
    with st.sidebar.expander("Optional drug coverage"):
        requested = st.text_input(
            "Additional requested drugs (comma-separated)",
            help="Uncovered names render as first-class no-call cards.",
        )
    additions = tuple(value.strip() for value in requested.split(",") if value.strip())
    drugs_of_interest = tuple(dict.fromkeys([*covered_drugs, *additions])) if additions else ()
    config = replace(
        base_config,
        covered_species=tuple(covered_species) if covered_species else base_config.covered_species,
        drugs_of_interest=drugs_of_interest,
        ood_threshold=(
            predictor.ood_threshold()
            if source == "Real pipeline" and predictor is not None
            else base_config.ood_threshold
        ),
    )

    species_text = ", ".join(covered_species) if covered_species else "Unavailable"
    drugs_text = ", ".join(covered_drugs) if covered_drugs else "Unavailable"
    st.sidebar.markdown(
        '<div class="sidebar-panel">'
        '<span class="sidebar-eyebrow">Live coverage</span>'
        f'<span class="sidebar-value"><strong>Species:</strong> {html.escape(species_text)}</span>'
        f'<span class="sidebar-value"><strong>Drugs:</strong> {html.escape(drugs_text)}</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.sidebar.expander("Decision configuration"):
        st.caption(config.tuned_on)
        st.markdown(
            "Uncertainty band: "
            f'<span class="mono">[{config.uncertainty_band_low:.2f}, {config.uncertainty_band_high:.2f}]</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
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


def _save_report_and_open(
    report: GenomeReport,
    config: DecisionConfig,
    source: str,
) -> None:
    st.session_state["clinical_report"] = report
    st.session_state["clinical_config"] = config
    st.session_state["report_source"] = source
    _open_view(VIEW_REPORT)


def render_analysis_page(
    source: str,
    predictor: Any,
    extractor: Any,
    config: DecisionConfig,
    species: str,
) -> None:
    """Collect one genome input; successful generation navigates to the report page."""
    st.markdown(
        '<div class="section-kicker">Individual genome</div>'
        '<h2 class="section-heading">Analyze a genome</h2>',
        unsafe_allow_html=True,
    )
    render_mode_banner(source)

    if source == "Mock (demo)":
        input_mode = st.radio("Input method", ["Demo case", "Upload FASTA"], horizontal=True)
        cases = demo_cases()
        selected_name = st.selectbox("Demo case", [case.name for case in cases])
        selected_case = next(case for case in cases if case.name == selected_name)
        st.caption(selected_case.description)
        selected_species = st.text_input("Report species", value=selected_case.species)
        upload = None
        if input_mode == "Upload FASTA":
            upload = st.file_uploader("FASTA file", type=["fasta", "fa", "fna", "fas"])

        if st.button("Generate clinical report", type="primary", use_container_width=True):
            try:
                with st.spinner("Running the decision pipeline over this genome..."):
                    if input_mode == "Demo case":
                        scripted = scripted_predictor_for(cases)
                        generated_report = build_report(
                            selected_case.features,
                            scripted,
                            config,
                            selected_species,
                        )
                    elif upload is not None:
                        generated_report = _run_uploaded_mock(
                            upload,
                            predictor,
                            extractor,
                            config,
                            selected_species,
                        )
                    else:
                        st.error("Choose a FASTA file before generating a report.")
                        return
                _save_report_and_open(generated_report, config, source)
            except (DecisionReportError, IntegrationError) as exc:
                st.session_state.pop("clinical_report", None)
                st.error(f"Decision pipeline unavailable: {exc}")
            except Exception as exc:
                st.session_state.pop("clinical_report", None)
                st.error(f"The decision pipeline could not complete: {exc}")
    else:
        store_ids: list[str] = []
        if extractor is not None and hasattr(extractor, "genome_ids"):
            try:
                store_ids = extractor.genome_ids()
            except Exception:  # never let a store hiccup blank the page
                store_ids = []
        if store_ids:
            st.write("Pick a genome present in the configured Module 1 feature store.")
            identifier = st.selectbox("Genome ID", store_ids)
        else:
            st.write("Enter an identifier already present in the configured Module 1 feature store.")
            identifier = st.text_input("Genome ID or original FASTA filename")
        if st.button("Generate clinical report", type="primary", use_container_width=True):
            if predictor is None or extractor is None:
                st.session_state.pop("clinical_report", None)
                st.error("Real pipeline is not available. Check the sidebar artifact paths.")
            elif not identifier.strip():
                st.session_state.pop("clinical_report", None)
                st.error("Enter a genome ID or FASTA filename before generating a report.")
            else:
                try:
                    with st.spinner("Running the decision pipeline over this genome..."):
                        features = extractor(identifier.strip())
                        generated_report = build_report(features, predictor, config, species)
                    _save_report_and_open(generated_report, config, source)
                except (DecisionReportError, IntegrationError) as exc:
                    st.session_state.pop("clinical_report", None)
                    st.error(f"Decision pipeline unavailable: {exc}")
                except Exception as exc:
                    st.session_state.pop("clinical_report", None)
                    st.error(f"The decision pipeline could not complete: {exc}")

    st.info(
        "Choose a demo case or provide an input, then generate a report. "
        "The finished report opens on its own page."
    )
    if predictor is not None:
        st.caption(
            "Active coverage: "
            + ", ".join(predictor.covered_species())
            + " · "
            + ", ".join(predictor.covered_drugs())
        )


def render_report_page() -> None:
    """Render only the saved report, without repeating the input workspace."""
    report: GenomeReport | None = st.session_state.get("clinical_report")
    config: DecisionConfig | None = st.session_state.get("clinical_config")
    source = st.session_state.get(
        "report_source",
        st.session_state.get("active_source", "Mock (demo)"),
    )

    if report is None or config is None:
        st.warning("No generated report is available yet.")
        if st.button("Go to analysis", type="primary"):
            _open_view(VIEW_ANALYZE)
        return

    render_mode_banner(source)
    render_report(report, config)


def _display_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "Not available"
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def render_evaluation_page() -> None:
    st.write("Mock held-out evaluation only. Aggregate metrics stay separate from individual genome reports.")
    input_columns = st.columns(2, gap="medium")
    with input_columns[0]:
        n = int(st.number_input("Held-out genomes", min_value=10, max_value=500, value=60, step=10))
    with input_columns[1]:
        seed = int(st.number_input("Random seed", min_value=0, value=20260718, step=1))
    if st.button("Run mock evaluation", use_container_width=True):
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
    st.dataframe(overall_frame, use_container_width=True, hide_index=True)
    no_call = result.overall.get("no_call_rate")
    if _finite_number(no_call):
        st.markdown(
            f'No-call rate: <span class="mono">{float(no_call):.3f}</span>. '
            "This is reported explicitly rather than treating abstentions as confident predictions.",
            unsafe_allow_html=True,
        )

    st.markdown("**Per-drug performance**")
    st.dataframe(result.per_drug, use_container_width=True, hide_index=True)
    st.markdown("**Performance by genetic group**")
    st.dataframe(result.per_group, use_container_width=True, hide_index=True)
    st.markdown("**Reliability by probability bin**")
    reliability = pd.DataFrame([asdict(item) for item in result.reliability])
    if reliability.empty:
        st.write("No reliability bins were available.")
    else:
        st.dataframe(reliability, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title=f"{PRODUCT_NAME} · {PRODUCT_LONG_NAME}",
        page_icon="🧬",
        layout="wide",
    )
    inject_styles()
    active_view = st.session_state.get("active_view", VIEW_ANALYZE)
    if active_view not in {VIEW_ANALYZE, VIEW_REPORT, VIEW_EVALUATION}:
        active_view = VIEW_ANALYZE
        st.session_state["active_view"] = active_view
    if active_view == VIEW_REPORT and st.session_state.get("clinical_report") is None:
        active_view = VIEW_ANALYZE
        st.session_state["active_view"] = active_view

    if active_view == VIEW_ANALYZE:
        render_hero()
    elif active_view == VIEW_REPORT:
        render_page_masthead(
            "Clinical report",
            "Generated result",
            "Review each antibiotic decision and open its evidence details as needed.",
        )
    else:
        render_page_masthead(
            "Evaluation",
            "Model quality",
            "Run the held-out mock panel without loading the genome-analysis workspace.",
        )

    render_disclaimer()
    render_navigation(active_view)

    if active_view == VIEW_ANALYZE:
        source, predictor, extractor, config, species = _sidebar()
        render_analysis_page(source, predictor, extractor, config, species)
    elif active_view == VIEW_REPORT:
        render_report_page()
    else:
        render_evaluation_page()


if __name__ == "__main__":
    main()
