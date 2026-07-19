"""
Stage 3 — SHAP global + local explainability + literature validation table.

Uses LinearExplainer (exact and fast for the LR winner). Waterfall plots for
one confident TP and one confident TN. Writes a CSV of feature importance
consumed downstream by rigor/reports.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hormobench import (
    CLEAN_CSV, SPLITS_JSON, MODEL_JOBLIB, PREPROC_JOBLIB,
    RESULTS_FIG, SHAP_CSV, DECISION_THRESHOLD,
)

TARGET = "PCOS (Y/N)"


def main():
    df       = pd.read_csv(CLEAN_CSV)
    splits   = json.load(open(SPLITS_JSON))
    model    = joblib.load(MODEL_JOBLIB)
    prep     = joblib.load(PREPROC_JOBLIB)

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    feature_names = X.columns.tolist()

    X_train_pp = prep.transform(X.loc[splits["train_idx"]])
    X_test_pp  = prep.transform(X.loc[splits["test_idx"]])
    y_test     = y.loc[splits["test_idx"]].values

    # ---- SHAP LinearExplainer (exact for LR) ----
    explainer = shap.LinearExplainer(model, X_train_pp, feature_names=feature_names)
    shap_values = explainer(X_test_pp)

    importance_df = (
        pd.DataFrame({
            "feature":       feature_names,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(SHAP_CSV, index=False)
    print(f"✓ SHAP importance table → {SHAP_CSV}")

    # ---- Global plot: bar + beeswarm ----
    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("HormoBench-PCOS: SHAP Global Explainability",
                 fontsize=15, fontweight="bold")
    plt.sca(axes[0]); shap.plots.bar(shap_values, max_display=15, show=False)
    axes[0].set_title("(a) Feature Importance (mean |SHAP|)")
    plt.sca(axes[1]); shap.plots.beeswarm(shap_values, max_display=15, show=False)
    axes[1].set_title("(b) Feature Impact Direction (higher value → red)")
    plt.tight_layout()
    plt.savefig(RESULTS_FIG / "shap_global.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Waterfall: one confident TP and one confident TN ----
    y_prob = model.predict_proba(X_test_pp)[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    tp_idx = np.where((y_pred == 1) & (y_test == 1))[0]
    tn_idx = np.where((y_pred == 0) & (y_test == 0))[0]
    tp_pick = tp_idx[np.argmax(y_prob[tp_idx])]
    tn_pick = tn_idx[np.argmin(y_prob[tn_idx])]

    for label, pick, path in [
        ("True Positive", tp_pick, "shap_waterfall_TP.png"),
        ("True Negative", tn_pick, "shap_waterfall_TN.png"),
    ]:
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_values[pick], max_display=12, show=False)
        plt.title(f"{label} — Predicted P(PCOS) = {y_prob[pick]:.2f}",
                  fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_FIG / path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"✓ Figures written → {RESULTS_FIG}/shap_*.png")


if __name__ == "__main__":
    main()
