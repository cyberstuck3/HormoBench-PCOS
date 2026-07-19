# Architecture Decision Records

Concise, timestamped rationale for the five methodological decisions that
most shape HormoBench-PCOS. Each record follows the pattern: *context →
decision → rationale → consequences → alternatives considered*.

## Index

| # | Decision | Status |
|---|----------|:------:|
| [ADR-001](ADR-001-logistic-regression-baseline.md) | Logistic Regression as the primary reference model | Accepted |
| [ADR-002](ADR-002-threshold-optimization.md)       | Optimise the decision threshold on the validation set | Accepted |
| [ADR-003](ADR-003-shap-linear-explainer.md)        | Use SHAP LinearExplainer (exact) rather than Kernel or Tree | Accepted |
| [ADR-004](ADR-004-platt-scaling-conditional.md)    | Apply Platt scaling only if raw ECE > 0.05 | Accepted |
| [ADR-005](ADR-005-bootstrap-confidence-intervals.md) | Report 1000-iteration bootstrap 95% CIs for every metric | Accepted |

All records dated 2026-07-19.
