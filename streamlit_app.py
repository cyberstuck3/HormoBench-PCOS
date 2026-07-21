"""
streamlit_app.py — Lightweight demo frontend for HormoBench-PCOS.

This file is a *demo-only* wrapper around the existing artifacts produced by
`python scripts/benchmark.py`. It does NOT retrain, re-tune, or recompute
anything. It only:

    • loads results/models/best_model.joblib
    • loads results/models/preprocessor.joblib
    • loads results/results.json           (best_model name + decision threshold)
    • loads results/rigor_all_results.json (test metrics with 95% CIs)
    • loads results/shap_importance.csv    (precomputed SHAP importance)
    • displays results/figures/*.png       (SHAP + ROC/PR/calibration/CV)

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Make the existing `hormobench` package importable without modifying it ---
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hormobench import (  # noqa: E402
    CLEAN_CSV,
    BASELINE_JSON,
    RIGOR_JSON,
    SHAP_CSV,
    MODEL_JOBLIB,
    PREPROC_JOBLIB,
    RESULTS_FIG,
    load_decision_threshold,
)

TARGET = "PCOS (Y/N)"
GITHUB_URL = "https://github.com/"  # replace if desired; kept generic

# ─────────────────────────────────────────────────────────────────────────────
# Navigation — single source of truth
# ─────────────────────────────────────────────────────────────────────────────
SECTIONS = [
    "Overview",
    "About",
    "Model Features",
    "Prediction",
    "Explainability",
    "Model Performance",
]


def _go(section: str):
    """Navigation callback. Mutates the sidebar-bound state key BEFORE the
    next render, so the radio widget picks up the new value and highlights
    it correctly in the same rerun. Using on_click callbacks is the
    documented Streamlit pattern for driving a widget's state from another
    widget without triggering a stale-highlight race."""
    st.session_state["section"] = section


def _reset_prediction():
    """Clear any stored prediction so the form is fresh."""
    st.session_state.pop("prediction_result", None)

# ─────────────────────────────────────────────────────────────────────────────
# Page config + styling
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HormoBench-PCOS · Explainable AI Benchmark",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background-color: #ffffff; }
      h1, h2, h3, h4 { color: #0b2545; font-family: 'Inter', 'Segoe UI', sans-serif; }
      .hero {
          background: linear-gradient(135deg, #eaf2ff 0%, #ffffff 100%);
          padding: 2.2rem 2rem;
          border-radius: 16px;
          border: 1px solid #dfe8f5;
          margin-bottom: 1.5rem;
      }
      .hero h1 { margin: 0 0 .3rem 0; font-size: 2.2rem; }
      .hero p.tag { color: #1d4ed8; font-weight: 600; margin: 0; }
      .hero p.desc { color: #33445c; margin-top: .8rem; font-size: 1.02rem; }
      .card {
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 14px;
          padding: 1.1rem 1.2rem;
          height: 100%;
          box-shadow: 0 1px 2px rgba(15,23,42,.03);
      }
      .card h4 { margin-top: 0; color: #1d4ed8; }
      .card p  { color: #475569; margin-bottom: 0; font-size: .95rem; }
      .pill {
          display: inline-block;
          padding: .25rem .6rem;
          border-radius: 999px;
          background: #eaf2ff;
          color: #1d4ed8;
          font-size: .78rem;
          font-weight: 600;
          margin-right: .35rem;
      }
      div[data-testid="stMetric"] {
          background: #f8fafc;
          border: 1px solid #e2e8f0;
          border-radius: 12px;
          padding: .6rem .8rem;
      }
      .footer { color:#64748b; font-size:.85rem; text-align:center; margin-top:2rem; }
      .result-card {
          border-radius: 16px;
          padding: 1.4rem 1.6rem;
          margin: .6rem 0 1rem 0;
          border: 1px solid transparent;
      }
      .result-card h3 { margin: 0 0 .3rem 0; }
      .result-card .prob { font-size: 2.4rem; font-weight: 700; letter-spacing: -.02em; }
      .result-card .sub  { opacity: .85; font-size: .95rem; }
      .result-positive {
          background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
          border-color: #fecaca;
          color: #7f1d1d;
      }
      .result-negative {
          background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);
          border-color: #a7f3d0;
          color: #065f46;
      }
      .result-borderline {
          background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
          border-color: #fde68a;
          color: #78350f;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders (all artifacts are precomputed on disk)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_JOBLIB)
    preprocessor = joblib.load(PREPROC_JOBLIB)
    return model, preprocessor


@st.cache_data(show_spinner=False)
def load_reference_frame() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_CSV)
    return df


@st.cache_data(show_spinner=False)
def load_results_json() -> dict:
    with open(BASELINE_JSON) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_rigor_json() -> dict:
    if RIGOR_JSON.exists():
        with open(RIGOR_JSON) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_shap_importance() -> pd.DataFrame | None:
    if SHAP_CSV.exists():
        return pd.read_csv(SHAP_CSV)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Load everything up-front
# ─────────────────────────────────────────────────────────────────────────────
try:
    model, preprocessor = load_artifacts()
    reference_df = load_reference_frame()
    results_json = load_results_json()
    rigor_json = load_rigor_json()
    shap_importance = load_shap_importance()
    threshold = load_decision_threshold()
except Exception as exc:
    st.error(
        f"Failed to load model artifacts.\n\n"
        f"Have you run `python scripts/benchmark.py` first?\n\n"
        f"Error: `{exc}`"
    )
    st.stop()

FEATURE_COLS: list[str] = [c for c in reference_df.columns if c != TARGET]
BEST_MODEL_NAME = results_json.get("best_model", "Logistic Regression")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🩺 HormoBench-PCOS")
st.sidebar.caption("Explainable AI benchmark for PCOS")

# Initialise the single navigation key BEFORE the widget is created.
# The radio's key="section" binds the widget directly to this key, so any
# callback that mutates st.session_state["section"] will update both the
# highlighted item in the sidebar AND the rendered page in the same rerun.
if "section" not in st.session_state:
    st.session_state["section"] = "Overview"

section = st.sidebar.radio(
    "Navigate",
    SECTIONS,
    key="section",
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Best model** &nbsp;·&nbsp; `{BEST_MODEL_NAME}`  \n"
    f"**Decision τ** &nbsp;·&nbsp; `{threshold:.2f}`  \n"
    f"**Features** &nbsp;·&nbsp; `{len(FEATURE_COLS)}`"
)
st.sidebar.markdown("---")
st.sidebar.caption("For research use only. Not a diagnostic tool.")


# ─────────────────────────────────────────────────────────────────────────────
# Section: Overview / Hero
# ─────────────────────────────────────────────────────────────────────────────
def render_hero():
    st.markdown(
        """
        <div class="hero">
          <p class="tag">Explainable Medical AI · Benchmark</p>
          <h1>HormoBench-PCOS</h1>
          <p class="desc">
            An open, reproducible, and explainable benchmark for
            <b>Polycystic Ovary Syndrome</b> risk prediction — combining a
            calibrated ML pipeline with SHAP-based clinical interpretability.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        st.button(
            "▶  Start Prediction",
            type="primary",
            use_container_width=True,
            on_click=_go,
            args=("Prediction",),
            key="hero_start_predict",
        )
    with c2:
        st.link_button("GitHub Repository", GITHUB_URL, use_container_width=True)

    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    tm = rigor_json.get("test_metrics", {})

    def _pt(k, fallback=None):
        v = tm.get(k, {})
        return v.get("point", fallback) if isinstance(v, dict) else fallback

    m1.metric("Test ROC AUC", f"{_pt('roc_auc', 0):.3f}" if _pt("roc_auc") else "—")
    m2.metric("Recall (Sensitivity)", f"{_pt('recall', 0):.3f}" if _pt("recall") else "—")
    m3.metric("Balanced Accuracy", f"{_pt('balanced_accuracy', 0):.3f}" if _pt("balanced_accuracy") else "—")
    m4.metric("Best Model", BEST_MODEL_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# Section: About
# ─────────────────────────────────────────────────────────────────────────────
def render_about():
    st.header("About")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            #### What is PCOS?
            Polycystic Ovary Syndrome is a common endocrine disorder affecting
            an estimated **8–13% of women of reproductive age**. It presents
            heterogeneously — hormonal imbalance, irregular cycles, hyperandrogenism,
            and polycystic ovarian morphology — which makes it easy to under-diagnose.
            """
        )
        st.markdown(
            """
            #### Why explainability matters
            In clinical AI, a probability alone is not enough. Practitioners need
            to know *why* a model flagged a case. HormoBench pairs every prediction
            with SHAP attributions so each risk score is traceable to specific
            clinical and hormonal features.
            """
        )
    with col2:
        st.markdown(
            """
            #### What HormoBench-PCOS does
            - Provides a **reproducible benchmark** across four ML baselines
              (Logistic Regression, SVM-RBF, KNN, Naive Bayes).
            - Reports **bootstrap 95% confidence intervals** on all test metrics
              and **DeLong tests** for AUC comparisons.
            - Publishes a **calibrated decision threshold** optimized against a
              recall-oriented objective on the validation split.
            - Ships **SHAP global + local** explanations and a
              **literature-alignment** check on the top drivers.
            - Everything runs from one command: `python scripts/benchmark.py`.
            """
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section: Model Features (cards)
# ─────────────────────────────────────────────────────────────────────────────
def render_features():
    st.header("Model Features")
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Explainable AI", "SHAP global + waterfall plots for every prediction — each risk score is traceable to specific features."),
        ("Machine Learning Benchmark", "Four calibrated baselines tuned with 5-fold stratified CV; leakage-safe preprocessing inside each fold."),
        ("SHAP Interpretation", "Precomputed feature importance aligned with the PCOS literature (LH, follicle counts, cycle irregularity, AMH)."),
        ("Reproducible Research", "Deterministic seeds, split manifest, model card, dataset card, and one-command reproduction."),
    ]
    for col, (title, body) in zip((c1, c2, c3, c4), cards):
        with col:
            st.markdown(
                f'<div class="card"><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Section: Prediction — dynamic form from the trained feature list
# ─────────────────────────────────────────────────────────────────────────────
def _is_binary_col(col: str, series: pd.Series) -> bool:
    if "(Y/N)" in col:
        return True
    vals = pd.Series(series.dropna().unique())
    if len(vals) <= 2 and set(vals.astype(float).unique()).issubset({0.0, 1.0}):
        return True
    return False


def _default(series: pd.Series) -> float:
    return float(series.median()) if len(series.dropna()) else 0.0


def render_prediction():
    st.header("Prediction")
    st.caption(
        f"Enter clinical values. Defaults are set to dataset medians — "
        f"you only need to adjust what you know. Model expects **{len(FEATURE_COLS)} features**."
    )

    # Rank features by SHAP importance so the most impactful inputs come first
    if shap_importance is not None:
        ranking = shap_importance.set_index("feature")["mean_abs_shap"].to_dict()
    else:
        ranking = {}
    ordered_cols = sorted(FEATURE_COLS, key=lambda c: -ranking.get(c, 0.0))

    top_cols = ordered_cols[:8]
    rest_cols = ordered_cols[8:]

    inputs: dict[str, float] = {}

    def _widget(col: str, key_prefix: str):
        series = reference_df[col]
        default = _default(series)
        if _is_binary_col(col, series):
            choice = st.radio(
                col,
                options=["No", "Yes"],
                index=int(default >= 0.5),
                horizontal=True,
                key=f"{key_prefix}_{col}",
            )
            inputs[col] = 1.0 if choice == "Yes" else 0.0
        else:
            lo = float(np.nanmin(series)) if series.notna().any() else 0.0
            hi = float(np.nanmax(series)) if series.notna().any() else 1.0
            step = max((hi - lo) / 200.0, 0.01)
            inputs[col] = st.number_input(
                col,
                value=float(default),
                min_value=float(min(lo, default)),
                max_value=float(max(hi, default) * 1.5 if hi > 0 else default + 1),
                step=step,
                format="%.2f",
                key=f"{key_prefix}_{col}",
            )

    st.markdown("#### Key clinical inputs")
    st.caption("Top 8 features by SHAP global importance.")
    grid = st.columns(2)
    for i, col in enumerate(top_cols):
        with grid[i % 2]:
            _widget(col, "top")

    with st.expander(f"Advanced clinical inputs ({len(rest_cols)} more features) — optional, defaults to medians"):
        grid2 = st.columns(2)
        for i, col in enumerate(rest_cols):
            with grid2[i % 2]:
                _widget(col, "adv")

    st.markdown("")
    predict_clicked = st.button(
        "🔬  Predict PCOS risk",
        type="primary",
        use_container_width=True,
        key="predict_submit",
    )

    if predict_clicked:
        # Brief animated status for perceived polish — actual inference is instant.
        import time
        with st.status("Running inference pipeline…", expanded=False) as status:
            st.write("Assembling patient feature vector…")
            row = pd.DataFrame([[inputs[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
            time.sleep(0.15)
            st.write("Applying preprocessing (median imputation + standardization)…")
            X = preprocessor.transform(row)
            time.sleep(0.15)
            st.write("Scoring with the tuned classifier…")
            proba = float(model.predict_proba(X)[0, 1])
            label = int(proba >= threshold)
            time.sleep(0.1)
            status.update(label="Inference complete", state="complete", expanded=False)

        # Persist result across reruns so navigation and follow-up buttons don't lose it.
        st.session_state["prediction_result"] = {
            "proba": proba,
            "label": label,
            "threshold": threshold,
        }

    # Render the result (if any) — driven by session_state so it survives reruns.
    result = st.session_state.get("prediction_result")
    if result:
        _render_prediction_result(result)


def _render_prediction_result(result: dict):
    proba = result["proba"]
    label = result["label"]
    tau = result["threshold"]
    margin = abs(proba - tau)
    confidence = margin / max(tau, 1 - tau)

    st.success("✓ Prediction generated successfully.")
    st.markdown("### Prediction Result")

    # Colored status card: green / red / amber depending on outcome + margin
    if margin < 0.05:
        css_class = "result-borderline"
        headline = "Borderline"
        emoji = "⚖️"
        subtitle = "Score is close to the decision threshold — review SHAP attributions carefully."
    elif label == 1:
        css_class = "result-positive"
        headline = "PCOS-positive"
        emoji = "🔴"
        subtitle = f"Predicted probability exceeds the calibrated threshold τ = {tau:.2f}."
    else:
        css_class = "result-negative"
        headline = "PCOS-negative"
        emoji = "🟢"
        subtitle = f"Predicted probability is below the calibrated threshold τ = {tau:.2f}."

    st.markdown(
        f"""
        <div class="result-card {css_class}">
          <h3>{emoji}&nbsp; {headline}</h3>
          <div class="prob">P(PCOS) = {proba*100:.1f}%</div>
          <div class="sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("PCOS probability", f"{proba*100:.1f}%")
    c2.metric("Decision threshold (τ)", f"{tau:.2f}")
    conf_label = "High" if confidence > 0.6 else "Moderate" if confidence > 0.3 else "Low"
    c3.metric("Confidence", conf_label, f"|P − τ| = {margin:.3f}", delta_color="off")

    st.progress(min(max(proba, 0.0), 1.0), text=f"P(PCOS) = {proba:.3f}")

    st.caption(
        "⚠️ Research and educational demo only. This is not a diagnostic device "
        "and must not be used for clinical decisions."
    )

    # Next-step navigation — advances the hackathon demo flow.
    st.markdown("#### Next")
    b1, b2, b3 = st.columns(3)
    b1.button(
        "🔁  Try Another Prediction",
        use_container_width=True,
        on_click=_reset_prediction,
        key="btn_try_another",
    )
    b2.button(
        "🧠  View SHAP Explanation",
        type="primary",
        use_container_width=True,
        on_click=_go,
        args=("Explainability",),
        key="btn_view_shap",
    )
    b3.button(
        "📊  View Model Performance",
        use_container_width=True,
        on_click=_go,
        args=("Model Performance",),
        key="btn_view_perf",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section: Explainability — reuse precomputed SHAP figures
# ─────────────────────────────────────────────────────────────────────────────
def render_explain():
    st.header("Explainability")
    st.caption(
        "Global and local SHAP explanations for the winning model. "
        "Generated once by `scripts/explain.py` — displayed here directly."
    )

    fig_global = RESULTS_FIG / "shap_global.png"
    fig_tp = RESULTS_FIG / "shap_waterfall_TP.png"
    fig_tn = RESULTS_FIG / "shap_waterfall_TN.png"

    if fig_global.exists():
        st.subheader("Global feature importance")
        st.image(str(fig_global), use_container_width=True)
    else:
        st.warning("`results/figures/shap_global.png` not found. Run `python scripts/explain.py`.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Local: confident True Positive")
        if fig_tp.exists():
            st.image(str(fig_tp), use_container_width=True)
        else:
            st.info("Waterfall (TP) not available.")
    with c2:
        st.subheader("Local: confident True Negative")
        if fig_tn.exists():
            st.image(str(fig_tn), use_container_width=True)
        else:
            st.info("Waterfall (TN) not available.")

    if shap_importance is not None:
        st.subheader("Top-10 features (mean |SHAP|)")
        top10 = shap_importance.head(10).copy()
        top10["mean_abs_shap"] = top10["mean_abs_shap"].round(4)
        st.dataframe(top10, use_container_width=True, hide_index=True)

        st.markdown(
            """
            **What the model is learning.** The top drivers align with the
            established PCOS phenotype — **LH**, bilateral **follicle counts**,
            metabolic signals (**weight gain**, **fast food**), and
            hyperandrogenic signs (**hair growth**, **skin darkening**,
            **pimples**) — supporting clinical face validity.
            """
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section: Model Performance
# ─────────────────────────────────────────────────────────────────────────────
def render_performance():
    st.header("Model Performance")
    st.caption(
        "Point estimates with bootstrap 95% confidence intervals on the held-out test set."
    )

    tm = rigor_json.get("test_metrics")
    if not tm:
        st.warning("`results/rigor_all_results.json` not found. Run `python scripts/run_rigor.py`.")
    else:
        def _row(keys):
            cols = st.columns(len(keys))
            labels = {
                "roc_auc": "ROC AUC",
                "pr_auc": "PR AUC",
                "precision": "Precision",
                "recall": "Recall",
                "specificity": "Specificity",
                "balanced_accuracy": "Balanced Acc.",
                "f1": "F1",
                "mcc": "MCC",
                "brier": "Brier score",
            }
            for c, k in zip(cols, keys):
                v = tm.get(k, {})
                if not v:
                    c.metric(labels.get(k, k), "—")
                    continue
                point = v.get("point")
                lo, hi = v.get("lo"), v.get("hi")
                delta = f"95% CI [{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else ""
                c.metric(labels.get(k, k), f"{point:.3f}", delta, delta_color="off")

        _row(["roc_auc", "pr_auc", "balanced_accuracy", "f1"])
        _row(["precision", "recall", "specificity", "mcc"])

    st.markdown("---")
    st.subheader("Diagnostic curves")
    cols = st.columns(2)
    fig_map = [
        ("ROC curves", "roc_curves.png"),
        ("Precision–Recall curves", "pr_curves.png"),
        ("Calibration", "calibration.png"),
        ("CV boxplots", "cv_boxplots.png"),
    ]
    for i, (title, fname) in enumerate(fig_map):
        path = RESULTS_FIG / fname
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info(f"`{fname}` not available yet.")


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
if section == "Overview":
    render_hero()
    st.markdown("---")
    render_features()
elif section == "About":
    render_about()
elif section == "Model Features":
    render_features()
elif section == "Prediction":
    render_prediction()
elif section == "Explainability":
    render_explain()
elif section == "Model Performance":
    render_performance()

st.markdown(
    '<p class="footer">HormoBench-PCOS · Explainable AI benchmark demo · Research use only</p>',
    unsafe_allow_html=True,
)
