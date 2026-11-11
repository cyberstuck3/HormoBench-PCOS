"""
HormoBench-PCOS
===============
Open, reproducible, explainable benchmark for PCOS risk prediction.

Standard paths are computed from this file's location so that scripts remain
portable across environments.
"""
from pathlib import Path

__version__ = "0.1.0"

# Repository root = two levels up from this file (src/hormobench/__init__.py)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Standard directories
DATA_RAW       = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_SPLITS    = REPO_ROOT / "data" / "splits"
RESULTS        = REPO_ROOT / "results"
RESULTS_FIG    = RESULTS / "figures"
RESULTS_TAB    = RESULTS / "tables"
RESULTS_NAR    = RESULTS / "narratives"
RESULTS_MOD    = RESULTS / "models"
DOCS           = REPO_ROOT / "docs"

# Standard file paths
CLEAN_CSV       = DATA_PROCESSED / "pcos_clean.csv"
SPLITS_JSON     = DATA_SPLITS / "splits.json"
BASELINE_JSON   = RESULTS / "results.json"
RIGOR_JSON      = RESULTS / "rigor_all_results.json"
SHAP_CSV        = RESULTS / "shap_importance.csv"
MODEL_JOBLIB    = RESULTS_MOD / "best_model.joblib"
PREPROC_JOBLIB  = RESULTS_MOD / "preprocessor.joblib"

# Metabolomics cohort track (separate cohort, separate split, separate results)
META_RAW_DIR    = DATA_RAW / "metabolomics"
META_CLEAN_CSV  = DATA_PROCESSED / "pcos_metabolomics_clean.csv"
META_SPLITS     = DATA_SPLITS / "metabolomics_splits.json"
META_RESULTS    = RESULTS / "metabolomics"

# Reproducibility constants
SEED               = 42
N_BOOTSTRAP        = 1000

# DECISION_THRESHOLD is NOT hardcoded here.  It is derived by Stage 5
# (scripts/optimize_threshold.py) from validation-set predictions and
# persisted into results/results.json.  Downstream scripts must always
# load it from there via load_decision_threshold().


def load_decision_threshold() -> float:
    """Load the decision threshold persisted by Stage 5.

    Falls back to 0.50 (sklearn default) if the key is missing — which
    means Stage 5 has not run yet.
    """
    import json
    with open(BASELINE_JSON) as f:
        data = json.load(f)
    return float(data.get("decision_threshold", 0.50))
