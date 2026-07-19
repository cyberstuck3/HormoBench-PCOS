# ADR-001: Logistic Regression as the Primary Reference Model

- **Status**: Accepted
- **Date**: 2026-07-19

## Context

The benchmark evaluates four candidate classifiers (Logistic Regression,
SVM RBF, KNN, Gaussian Naive Bayes). One must be designated the *primary
reference model* — the one whose test-set metrics, calibration curve, and
SHAP attributions anchor the paper-style write-up.

## Decision

**Logistic Regression with L2 penalty (C = 0.01)** is the primary
reference model.

## Rationale

- **No statistically significant loss**. DeLong pairwise tests show that
  LR is not significantly better than SVM (p = 0.226), KNN (p = 0.192),
  or Naive Bayes (p = 0.072) at α = 0.05. Choosing the simpler model
  therefore does not cost measurable accuracy.
- **Exact explainability**. `shap.LinearExplainer` provides exact SHAP
  values for linear models in milliseconds, without the sampling
  approximation of `KernelExplainer` or the tree-structure assumptions
  of `TreeExplainer`. See [ADR-003](ADR-003-shap-linear-explainer.md).
- **Interpretable coefficients**. Sign and magnitude of each feature's
  weight can be inspected directly by a clinician.
- **Smallest overfitting gap**. Train ROC-AUC 0.968 vs test ROC-AUC 0.951
  (gap = 0.017) is the tightest of the four candidates.
- **Cheapest to reproduce**. Full 5-fold CV grid search on 377 training
  samples completes in seconds on a CPU.

## Consequences

- **Positive**: interpretability, reproducibility, cheap to run, invites
  extension.
- **Negative**: forfeits the ~1% point-estimate gain a tree ensemble
  might yield; unable to model non-linear feature interactions unless
  interaction terms are added by hand.

## Alternatives Considered

- **SVM (RBF)** — comparable ROC-AUC but requires KernelExplainer for
  SHAP, which is 100–1000× slower and introduces sampling error.
- **KNN** — competitive on ROC-AUC but no meaningful feature
  attribution; PR-AUC lags noticeably.
- **Naive Bayes** — fastest to train but the weakest on PR-AUC
  (0.822 ± 0.048) and the assumption of feature independence is
  clinically implausible for hormonal panels.
- **Gradient-boosted trees (XGBoost, LightGBM)** — likely to gain 1–3
  points of ROC-AUC. Rejected because (i) the marginal gain does not
  justify the loss of interpretability, (ii) tree ensembles complicate
  the reproducibility contract by adding another dependency and its
  own random-state handling, and (iii) the challenge briefing values
  transparency over sheer size.

## References

- Sun & Xu (2014), IEEE SPL 21(11). Fast DeLong implementation.
- Full comparative results: [`docs/evaluation_section.md`](../evaluation_section.md) § 2.
