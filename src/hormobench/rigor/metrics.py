"""Comprehensive metrics with bootstrap 95% confidence intervals."""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix,
)


def specificity(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


PROB_METRICS = {  # take probabilities
    'roc_auc': roc_auc_score,
    'pr_auc':  average_precision_score,
    'brier':   brier_score_loss,
}

PRED_METRICS = {  # take hard predictions
    'precision':         lambda yt, yp: precision_score(yt, yp, zero_division=0),
    'recall':            lambda yt, yp: recall_score(yt, yp, zero_division=0),
    'specificity':       specificity,
    'f1':                lambda yt, yp: f1_score(yt, yp, zero_division=0),
    'balanced_accuracy': balanced_accuracy_score,
    'mcc':               matthews_corrcoef,
}


def bootstrap_ci(y_true, values, metric_fn, n_iter=1000, seed=42):
    """Non-parametric bootstrap 95% CI (percentile method)."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    values = np.asarray(values)
    n = len(y_true)
    scores = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        yt, vv = y_true[idx], values[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(metric_fn(yt, vv))
        except Exception:
            continue
    scores = np.asarray(scores)
    point = metric_fn(y_true, values)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {
        'point':   float(point),
        'lo':      float(lo),
        'hi':      float(hi),
        'std':     float(scores.std()),
        'n_valid': int(len(scores)),
    }


def compute_all_metrics(y_true, y_prob, threshold=0.5, n_iter=1000, seed=42):
    """Compute all metrics with bootstrap CIs at the given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    for name, fn in PROB_METRICS.items():
        out[name] = bootstrap_ci(y_true, y_prob, fn, n_iter=n_iter, seed=seed)
    for name, fn in PRED_METRICS.items():
        out[name] = bootstrap_ci(y_true, y_pred, fn, n_iter=n_iter, seed=seed)
    return out
