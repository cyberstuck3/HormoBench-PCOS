# ADR-003: Use SHAP LinearExplainer (Exact) Rather Than Kernel or Tree

- **Status**: Accepted
- **Date**: 2026-07-19

## Context

SHAP offers several explainers with different trade-offs between speed,
generality, and exactness. Because the primary reference model is
Logistic Regression, the choice is not neutral.

## Decision

Use `shap.LinearExplainer` for all SHAP computations in this benchmark.

## Rationale

- **Exact for linear models**. For LR, LinearExplainer computes the
  closed-form Shapley values with no sampling; results are deterministic
  given the same background dataset.
- **Fast**. Global attributions over the full test set complete in under
  one second on a single CPU core. This keeps the total benchmark
  runtime under 20 seconds and encourages iterative research.
- **Independence-of-features vs correlation modes are configurable** —
  the default `feature_perturbation="interventional"` correctly handles
  correlated features (e.g. bilateral follicle counts) by breaking the
  correlation, giving each feature its individual contribution.
- **Downstream artefacts are stable**. `results/shap_importance.csv` is
  bit-identical run to run, which is critical for the reproducibility
  contract in [`RESULT_SCHEMA.md`](../RESULT_SCHEMA.md).

## Consequences

- **Positive**: reproducible, fast, exact, low-code footprint.
- **Negative**: tied to the choice of a linear reference model
  ([ADR-001](ADR-001-logistic-regression-baseline.md)). If a future
  version of the benchmark adds a tree or ensemble model as the
  reference, this ADR must be revisited.

## Alternatives Considered

- **KernelExplainer** — model-agnostic but 100–1000× slower and
  introduces sampling variance; would break bit-identical outputs.
- **TreeExplainer** — irrelevant given LR is the primary model.
- **Permutation importance from sklearn** — faster still but not
  additive over predictions, so cannot generate waterfall plots for
  individual patients.
- **DeepExplainer / GradientExplainer** — for neural nets, out of scope.

## References

- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to
  interpreting model predictions. *NeurIPS*.
- SHAP LinearExplainer docs:
  https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html
