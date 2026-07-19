# ADR-004: Apply Platt Scaling Only If Raw ECE > 0.05

- **Status**: Accepted
- **Date**: 2026-07-19

## Context

Well-calibrated probabilities matter for clinical decision-making —
downstream stakeholders (physicians, guideline authors) treat a "70 %
risk" score as an actual frequency. Uncalibrated probabilities mislead.
But applying calibration methods unconditionally is over-engineering
when the raw model is already well-calibrated, and can even worsen
calibration in small-sample regimes.

## Decision

Measure raw calibration first. Fit **Platt (sigmoid) scaling on the
validation set** and apply it to test predictions if and only if raw
ECE exceeds 0.05. Otherwise, report raw calibration only.

## Rationale

- **Measure before acting**. Blind application of Platt or isotonic
  scaling can degrade already-calibrated models, especially on
  small validation sets.
- **Threshold of 0.05 is a defensible norm** in medical-AI calibration
  reporting (Guo et al., 2017; Van Calster et al., 2019).
- **Empirically triggered**. The raw LR model on this dataset has
  ECE = 0.159 — strong regularisation (C = 0.01) makes it
  under-confident. Platt scaling reduces ECE to 0.080 and Brier from
  0.095 to 0.076. Applying it is the correct call.
- **Val set, not train, for fitting**. Model predictions on train are
  overfit; validation predictions give an unbiased signal for
  calibration.

## Consequences

- **Positive**: reported probabilities are trustworthy; the decision
  is data-driven, not habit.
- **Negative**: adds one Platt regression on top of the primary model,
  slightly increasing prediction latency (negligible for LR).

## Alternatives Considered

- **Always apply Platt** — rejected: unnecessary if not needed and
  may worsen small-sample calibration.
- **Isotonic regression** — more flexible but needs more data than 82
  validation samples to be reliable; Platt is preferred at this
  sample size.
- **Never calibrate; report raw only** — rejected because the raw
  model here is measurably miscalibrated.
- **Temperature scaling** — a neural-net idiom; overkill for a linear
  model.

## References

- Platt, J. (1999). Probabilistic outputs for support vector machines.
  *Advances in Large Margin Classifiers*.
- Guo, C. et al. (2017). On calibration of modern neural networks.
  *ICML*.
- Van Calster, B. et al. (2019). Calibration: the Achilles heel of
  predictive analytics. *BMC Medicine*, 17, 230.
