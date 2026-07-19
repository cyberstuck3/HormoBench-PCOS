"""
Deterministic tests for the HormoBench-PCOS rigor modules.

All tests use small, synthetic data with fixed seeds so they execute
in under a second and produce identical results on every run.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def binary_data():
    """Small deterministic binary classification data."""
    rng = np.random.RandomState(0)
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05])
    return y_true, y_prob


@pytest.fixture
def larger_data():
    """Larger deterministic data for bootstrap stability."""
    rng = np.random.RandomState(42)
    n = 100
    y_true = np.concatenate([np.ones(30), np.zeros(70)]).astype(int)
    y_prob = rng.beta(2, 5, n)
    # Make positives tend higher
    y_prob[y_true == 1] += 0.3
    y_prob = np.clip(y_prob, 0, 1)
    return y_true, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# C1: Threshold optimisation
# ─────────────────────────────────────────────────────────────────────────────

class TestThresholdOptimisation:
    def test_returns_valid_threshold(self, binary_data):
        """Threshold is in (0, 1) and recall is maximised above precision floor."""
        # Inline the function to avoid needing the full script
        from sklearn.metrics import precision_score, recall_score

        y_true, y_prob = binary_data
        grid = np.arange(0.05, 0.96, 0.01)
        precision_floor = 0.65

        best_tau, best_recall = 0.50, 0.0
        for tau in grid:
            y_pred = (y_prob >= tau).astype(int)
            if y_pred.sum() == 0:
                continue
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            if prec >= precision_floor and rec > best_recall:
                best_tau, best_recall = float(tau), rec

        assert 0.0 < best_tau < 1.0
        assert best_recall > 0.0

    def test_precision_floor_respected(self, binary_data):
        """Every threshold candidate that beats best_recall also meets precision floor."""
        from sklearn.metrics import precision_score, recall_score

        y_true, y_prob = binary_data
        grid = np.arange(0.05, 0.96, 0.01)
        floor = 0.65

        best_tau = 0.50
        best_recall = 0.0
        for tau in grid:
            y_pred = (y_prob >= tau).astype(int)
            if y_pred.sum() == 0:
                continue
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            if prec >= floor and rec > best_recall:
                best_tau, best_recall = float(tau), rec

        # Verify the winner meets the floor
        y_pred_best = (y_prob >= best_tau).astype(int)
        assert precision_score(y_true, y_pred_best, zero_division=0) >= floor

    def test_deterministic(self, binary_data):
        """Same input → same output every time."""
        from sklearn.metrics import precision_score, recall_score

        y_true, y_prob = binary_data
        results = []
        for _ in range(3):
            grid = np.arange(0.05, 0.96, 0.01)
            best_tau, best_rec = 0.50, 0.0
            for tau in grid:
                yp = (y_prob >= tau).astype(int)
                if yp.sum() == 0:
                    continue
                prec = precision_score(y_true, yp, zero_division=0)
                rec  = recall_score(y_true, yp, zero_division=0)
                if prec >= 0.65 and rec > best_rec:
                    best_tau, best_rec = float(tau), rec
            results.append(round(best_tau, 4))
        assert results[0] == results[1] == results[2]


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_returns_correct_keys(self, binary_data):
        from hormobench.rigor.metrics import bootstrap_ci
        from sklearn.metrics import roc_auc_score

        y_true, y_prob = binary_data
        result = bootstrap_ci(y_true, y_prob, roc_auc_score, n_iter=50, seed=42)
        assert set(result.keys()) == {"point", "lo", "hi", "std", "n_valid"}

    def test_ci_contains_point(self, binary_data):
        from hormobench.rigor.metrics import bootstrap_ci
        from sklearn.metrics import roc_auc_score

        y_true, y_prob = binary_data
        result = bootstrap_ci(y_true, y_prob, roc_auc_score, n_iter=200, seed=42)
        assert result["lo"] <= result["point"] <= result["hi"]

    def test_deterministic(self, larger_data):
        from hormobench.rigor.metrics import bootstrap_ci
        from sklearn.metrics import roc_auc_score

        y_true, y_prob = larger_data
        r1 = bootstrap_ci(y_true, y_prob, roc_auc_score, n_iter=100, seed=42)
        r2 = bootstrap_ci(y_true, y_prob, roc_auc_score, n_iter=100, seed=42)
        assert r1["point"] == r2["point"]
        assert r1["lo"] == r2["lo"]
        assert r1["hi"] == r2["hi"]

    def test_compute_all_metrics(self, larger_data):
        from hormobench.rigor.metrics import compute_all_metrics

        y_true, y_prob = larger_data
        result = compute_all_metrics(y_true, y_prob, threshold=0.5,
                                      n_iter=50, seed=42)
        expected_keys = {"roc_auc", "pr_auc", "brier",
                         "precision", "recall", "specificity",
                         "f1", "balanced_accuracy", "mcc"}
        assert expected_keys == set(result.keys())


# ─────────────────────────────────────────────────────────────────────────────
# DeLong test
# ─────────────────────────────────────────────────────────────────────────────

class TestDeLong:
    def test_identical_models_zero_delta(self, binary_data):
        from hormobench.rigor.delong import delong_pairwise

        y_true, y_prob = binary_data
        scores = {"A": y_prob, "B": y_prob}
        pw, aucs = delong_pairwise(y_true, scores)
        assert abs(pw["A vs B"]["delta"]) < 1e-10
        assert pw["A vs B"]["p_value"] >= 0.99

    def test_auc_close_to_sklearn(self, binary_data):
        from hormobench.rigor.delong import delong_pairwise
        from sklearn.metrics import roc_auc_score

        y_true, y_prob = binary_data
        scores = {"M": y_prob}
        _, aucs = delong_pairwise(y_true, scores)
        sklearn_auc = roc_auc_score(y_true, y_prob)
        assert abs(aucs["M"] - sklearn_auc) < 1e-6

    def test_output_structure(self, binary_data):
        from hormobench.rigor.delong import delong_pairwise

        y_true, y_prob = binary_data
        rng = np.random.RandomState(1)
        y_prob2 = y_prob + rng.normal(0, 0.05, len(y_prob))
        y_prob2 = np.clip(y_prob2, 0, 1)
        scores = {"A": y_prob, "B": y_prob2}
        pw, aucs = delong_pairwise(y_true, scores)
        assert "A vs B" in pw
        for key in ["auc_a", "auc_b", "delta", "se", "z", "p_value", "significant_at_0.05"]:
            assert key in pw["A vs B"]

    def test_deterministic(self, binary_data):
        from hormobench.rigor.delong import delong_pairwise

        y_true, y_prob = binary_data
        scores = {"A": y_prob, "B": y_prob * 0.9}
        pw1, _ = delong_pairwise(y_true, scores)
        pw2, _ = delong_pairwise(y_true, scores)
        assert pw1["A vs B"]["p_value"] == pw2["A vs B"]["p_value"]


# ─────────────────────────────────────────────────────────────────────────────
# Expected Calibration Error
# ─────────────────────────────────────────────────────────────────────────────

class TestECE:
    def test_perfect_calibration(self):
        from hormobench.rigor.calibration import expected_calibration_error

        # Probabilities = labels → perfectly calibrated → ECE ≈ 0
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ece = expected_calibration_error(y_true, y_prob, n_bins=2)
        assert ece < 0.01

    def test_bad_calibration_high_ece(self):
        from hormobench.rigor.calibration import expected_calibration_error

        # All predictions are 0.9 but only 50% are positive → poorly calibrated
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9] * 10)
        ece = expected_calibration_error(y_true, y_prob, n_bins=5)
        assert ece > 0.3

    def test_ece_in_range(self, larger_data):
        from hormobench.rigor.calibration import expected_calibration_error

        y_true, y_prob = larger_data
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_deterministic(self, larger_data):
        from hormobench.rigor.calibration import expected_calibration_error

        y_true, y_prob = larger_data
        ece1 = expected_calibration_error(y_true, y_prob, n_bins=10)
        ece2 = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece1 == ece2
