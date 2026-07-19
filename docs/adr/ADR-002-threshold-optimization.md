# ADR-002: Optimise the Decision Threshold on the Validation Set

- **Status**: Accepted
- **Date**: 2026-07-19

## Context

Every probabilistic classifier requires a threshold to convert probability
into a class label. sklearn defaults to 0.5, but 0.5 is a mathematical
convenience, not a clinical decision. For a screening task where missed
positives cause disproportionate harm, the default is not neutral.

## Decision

Select the decision threshold on the **validation set** by maximising
recall subject to a precision floor of 0.65, yielding **τ = 0.26**.
The test set is not consulted during threshold selection.

## Rationale

- **Asymmetric costs**. A missed PCOS diagnosis carries documented
  harms — sub-fertility, endometrial hyperplasia, 2–4× baseline type-2
  diabetes risk, cardiovascular morbidity (Teede et al., 2018). A false
  positive costs a single confirmatory visit applying the Rotterdam
  work-up.
- **Screening precedent**. Established screening programmes such as
  mammography and cervical cytology are calibrated toward high
  sensitivity for exactly this reason: the diagnostic pathway absorbs
  false positives cheaply, whereas false negatives leave disease
  undetected.
- **Empirical impact**. Moving from τ = 0.50 to τ = 0.26 changes the
  test-set confusion matrix from (9 FN, 2 FP) to (1 FN, 12 FP): an 89 %
  reduction in missed cases at the cost of 10 additional referrals.

## Consequences

- **Positive**: recall 0.963, sensitivity in line with medical screening
  norms, PR-AUC unchanged (threshold-independent), clinically
  defensible choice.
- **Negative**: precision drops to 0.68 — the model over-refers. In
  contexts where confirmatory testing is expensive or unavailable, a
  higher threshold would be preferable.

## Alternatives Considered

- **Default 0.5** — rejected on clinical grounds (see above).
- **Youden's J (max sensitivity + specificity − 1)** — considered; would
  yield τ ≈ 0.44 with recall 0.85. Rejected because it treats sensitivity
  and specificity symmetrically, which is inappropriate for screening.
- **Cost-sensitive optimisation** with explicit FN:FP ratio — attractive
  but requires eliciting the ratio, which varies by healthcare context.
  Deferred to a future refinement.
- **Threshold selection on the test set** — rejected as textbook data
  leakage.

## References

- Teede HJ et al. (2018). International evidence-based guideline for PCOS.
  *Human Reproduction* 33(9), 1602–1618.
- Full threshold analysis: [`results/narratives/threshold_justification.md`](../../results/narratives/threshold_justification.md)
