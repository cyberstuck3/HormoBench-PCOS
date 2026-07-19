"""Calibration analysis: Brier, ECE, and Platt scaling (applied only if ECE > 0.05)."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


ECE_THRESHOLD_FOR_PLATT = 0.05  # only apply Platt if ECE exceeds this


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE with equal-frequency (quantile) binning."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y_prob, quantiles)
    edges[0], edges[-1] = 0.0, 1.0 + 1e-10
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def calibration_analysis(y_true, y_prob, n_bins=10):
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='quantile'
    )
    return {
        'brier':     float(brier),
        'ece':       float(ece),
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist(),
    }


def apply_platt_scaling(base_model, X_val_pp, y_val, X_test_pp):
    """Fit Platt (sigmoid) on validation predictions, apply to test.
    Handles sklearn API changes: newer versions require FrozenEstimator,
    older versions accept cv='prefit'."""
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn ≥ 1.6
        cal = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model), method='sigmoid',
        )
    except ImportError:
        try:
            cal = CalibratedClassifierCV(
                estimator=base_model, method='sigmoid', cv='prefit',
            )
        except TypeError:  # sklearn < 1.2
            cal = CalibratedClassifierCV(
                base_estimator=base_model, method='sigmoid', cv='prefit',
            )
    cal.fit(X_val_pp, y_val)
    return cal.predict_proba(X_test_pp)[:, 1], cal
