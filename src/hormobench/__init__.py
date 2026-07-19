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

# Reproducibility constants
SEED               = 42
DECISION_THRESHOLD = 0.26   # tuned on validation set
N_BOOTSTRAP        = 1000
