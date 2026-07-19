"""Cross-validation mean ± SD across all candidate models."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def make_pipeline(clf):
    """Preprocessing + classifier — preprocessing is refit inside each CV fold."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf',     clf),
    ])


def cv_compare_models(models_dict, X_train, y_train, seed=42, n_splits=5):
    """
    models_dict : {name: unfitted sklearn classifier}
    X_train, y_train : raw (not preprocessed)
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for name, clf in models_dict.items():
        pipe = make_pipeline(clf)
        for metric in ['roc_auc', 'average_precision']:
            scores = cross_val_score(
                pipe, X_train, y_train, cv=cv, scoring=metric, n_jobs=-1
            )
            rows.append({
                'model':   name,
                'metric':  metric,
                'mean':    float(scores.mean()),
                'std':     float(scores.std()),
                'per_fold': scores.tolist(),
            })
    return pd.DataFrame(rows)
