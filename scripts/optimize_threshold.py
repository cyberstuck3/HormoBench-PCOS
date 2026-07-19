"""
Stage 5 — Threshold optimisation on the validation set.

Selects the decision threshold that maximises recall subject to a configurable
precision floor.  The chosen threshold is persisted into results/results.json
so that all downstream scripts (explain, run_rigor) load it from there instead
of recomputing or using a hardcoded constant.

The search grid, precision floor, and scoring objective are all configurable
via the constants below.
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
from sklearn.metrics import precision_score, recall_score

from hormobench import (
    CLEAN_CSV, SPLITS_JSON, BASELINE_JSON,
    MODEL_JOBLIB, PREPROC_JOBLIB, SEED,
)

TARGET = "PCOS (Y/N)"

# ── Configurable parameters ──────────────────────────────────────────────────
PRECISION_FLOOR   = 0.65          # minimum acceptable precision
THRESHOLD_GRID    = np.arange(0.05, 0.96, 0.01)  # search grid
# ─────────────────────────────────────────────────────────────────────────────


def find_optimal_threshold(y_true, y_prob, *,
                           precision_floor=PRECISION_FLOOR,
                           grid=THRESHOLD_GRID):
    """Maximise recall subject to precision >= precision_floor.

    Returns (best_threshold, best_recall, best_precision).
    Falls back to 0.50 if no threshold meets the precision floor.
    """
    best_tau, best_recall, best_precision = 0.50, 0.0, 0.0
    for tau in grid:
        y_pred = (y_prob >= tau).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        if prec >= precision_floor and rec > best_recall:
            best_tau, best_recall, best_precision = float(tau), rec, prec
    return round(best_tau, 4), best_recall, best_precision


def main():
    df      = pd.read_csv(CLEAN_CSV)
    splits  = json.load(open(SPLITS_JSON))
    model   = joblib.load(MODEL_JOBLIB)
    prep    = joblib.load(PREPROC_JOBLIB)

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_val_pp = prep.transform(X.loc[splits["val_idx"]])
    y_val    = y.loc[splits["val_idx"]].values

    y_val_prob = model.predict_proba(X_val_pp)[:, 1]

    tau, rec, prec = find_optimal_threshold(y_val, y_val_prob)

    print("=" * 60)
    print("THRESHOLD OPTIMISATION  (validation set only)")
    print("=" * 60)
    print(f"  Precision floor  : {PRECISION_FLOOR}")
    print(f"  Grid             : {THRESHOLD_GRID[0]:.2f} – {THRESHOLD_GRID[-1]:.2f} "
          f"(step {THRESHOLD_GRID[1] - THRESHOLD_GRID[0]:.2f})")
    print(f"  Optimal τ        : {tau}")
    print(f"  Val Recall @ τ   : {rec:.4f}")
    print(f"  Val Precision @ τ: {prec:.4f}")

    # ── Persist into results.json ────────────────────────────────────────────
    results = json.load(open(BASELINE_JSON))
    results["decision_threshold"] = tau
    results["threshold_config"] = {
        "precision_floor":    PRECISION_FLOOR,
        "grid_start":         float(THRESHOLD_GRID[0]),
        "grid_stop":          float(THRESHOLD_GRID[-1]),
        "grid_step":          float(THRESHOLD_GRID[1] - THRESHOLD_GRID[0]),
        "val_recall_at_tau":  rec,
        "val_precision_at_tau": prec,
    }
    json.dump(results, open(BASELINE_JSON, "w"), indent=2)
    print(f"\n✓ Threshold τ={tau} persisted → {BASELINE_JSON}")


if __name__ == "__main__":
    main()
