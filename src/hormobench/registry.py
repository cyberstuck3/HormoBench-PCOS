"""
Shared model registry — single source of truth for model builders, parameter
grids, and SHAP explainer dispatch.

Every script that needs a model definition must import from here instead of
defining its own.  This eliminates the duplication between train_baseline.py
and run_rigor.py and guarantees that the same model class + defaults are used
everywhere.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from hormobench import SEED


# ─────────────────────────────────────────────────────────────────────────────
# MODEL_REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# Each entry defines:
#   cls        — the sklearn estimator class
#   base_kw    — default kwargs applied on every instantiation
#   param_grid — GridSearchCV parameter space
#
# To add a new candidate model, add a single entry here. Every downstream
# script (train_baseline, run_rigor, explain) will pick it up automatically.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "Logistic Regression": {
        "cls":        LogisticRegression,
        "base_kw":    {"max_iter": 1000, "random_state": SEED},
        "param_grid": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]},
    },
    "SVM (RBF)": {
        "cls":        SVC,
        "base_kw":    {"kernel": "rbf", "probability": True, "random_state": SEED},
        "param_grid": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.001]},
    },
    "KNN": {
        "cls":        KNeighborsClassifier,
        "base_kw":    {},
        "param_grid": {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]},
    },
    "Naive Bayes": {
        "cls":        GaussianNB,
        "base_kw":    {},
        "param_grid": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
    },
}


def build_model(name: str, tuned_params: dict | None = None):
    """Instantiate a model from the registry, optionally merging tuned params."""
    entry = MODEL_REGISTRY[name]
    kw = {**entry["base_kw"]}
    if tuned_params:
        kw.update(tuned_params)
    return entry["cls"](**kw)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP explainer dispatch — C2
# ─────────────────────────────────────────────────────────────────────────────
_LINEAR_CLASSES = (LogisticRegression,)

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        AdaBoostClassifier, ExtraTreesClassifier,
    )
    _TREE_CLASSES = (
        DecisionTreeClassifier, RandomForestClassifier,
        GradientBoostingClassifier, AdaBoostClassifier,
        ExtraTreesClassifier,
    )
except ImportError:
    _TREE_CLASSES = ()

# Extend if xgboost / lightgbm are ever added
try:
    import xgboost
    _TREE_CLASSES = (*_TREE_CLASSES, xgboost.XGBClassifier)
except ImportError:
    pass
try:
    import lightgbm
    _TREE_CLASSES = (*_TREE_CLASSES, lightgbm.LGBMClassifier)
except ImportError:
    pass


def get_shap_explainer(model, X_background, feature_names=None):
    """Return the correct SHAP explainer for the given model type.

    Linear models  → shap.LinearExplainer  (exact, fast)
    Tree models    → shap.TreeExplainer    (exact for trees)
    Everything else → shap.KernelExplainer  (model-agnostic, slower)
    """
    import shap  # deferred so registry is importable without shap installed

    if isinstance(model, _LINEAR_CLASSES):
        return shap.LinearExplainer(model, X_background,
                                    feature_names=feature_names)
    if isinstance(model, _TREE_CLASSES):
        return shap.TreeExplainer(model)
    # Fallback: model-agnostic
    return shap.KernelExplainer(model.predict_proba, X_background)
