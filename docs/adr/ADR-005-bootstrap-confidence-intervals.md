# ADR-005: Report 1000-Iteration Bootstrap 95% CIs for Every Metric

- **Status**: Accepted
- **Date**: 2026-07-19

## Context

The test set contains 82 patients, of which 27 are PCOS-positive. Point
estimates on samples this small are noisy — an ROC-AUC of 0.951 is
compatible with true performance anywhere from the high 0.80s to nearly
1.0. Medical-AI reporting standards (TRIPOD-AI, DECIDE-AI) call for
uncertainty quantification alongside every point estimate.

## Decision

Report **95 % percentile bootstrap confidence intervals** based on 1000
resamples for every metric — threshold-independent (ROC-AUC, PR-AUC,
Brier) and threshold-dependent (precision, recall, F1, specificity,
balanced accuracy, MCC).

## Rationale

- **Non-parametric**. Bootstrap makes no distributional assumption,
  which is appropriate for metrics like MCC and F1 whose sampling
  distributions are analytically awkward.
- **1000 iterations** is the standard bar in TRIPOD-AI–aligned papers;
  sufficient for stable 2.5 / 97.5 percentiles on n = 82.
- **Percentile method** is the simplest defensible interval and does
  not require estimating the metric's variance analytically.
- **Reproducible**. The bootstrap seed is fixed (42), so CIs are
  bit-identical across runs.

## Consequences

- **Positive**: reported intervals discourage over-confidence in a
  single point estimate; readers can see width and asymmetry.
- **Negative**: 1000 iterations × 9 metrics adds ~1 second of runtime.
  Acceptable.

## Alternatives Considered

- **Point estimates only** — the norm in most hackathon submissions
  and much clinical-ML literature. Rejected as scientifically weak.
- **BCa (bias-corrected accelerated) bootstrap** — better in principle
  for asymmetric distributions but requires additional jackknife
  computation and is harder to explain. Overkill at 1000 iterations
  on 82 samples.
- **Analytical DeLong CIs for ROC-AUC** — DeLong is used for
  significance testing, not for the primary CI. The bootstrap gives
  us a uniform treatment across all metrics.
- **Bayesian credible intervals** — would require specifying priors
  and adds complexity; deferred.

## References

- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
- Collins, G. S. et al. (2024). TRIPOD-AI reporting guideline. *BMJ*.
