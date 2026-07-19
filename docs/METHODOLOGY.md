# Methodology

> **Purpose of this document.** Technical deep-dive on the design decisions
> that constitute HormoBench-PCOS as a scientifically rigorous benchmark.
> Complements [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) (formal spec) and the
> [ADRs](adr/) (decision records).

## 1. Anti-Leakage Architecture

Data leakage — allowing information from the evaluation set to influence
model or hyperparameter selection — is the single most common failure
mode in medical-AI benchmarks. HormoBench-PCOS enforces the following
invariants:

### 1.1 Preprocessing

Median imputation and standard scaling are wrapped in an sklearn
`Pipeline`. The pipeline is **refit inside every CV fold** (Stage 4)
and separately fit on the training set only (Stage 3) for the final
model. The validation and test sets are only ever *transformed*, never
*fit*. This ensures no median or standard-deviation value from
val/test contaminates the model.

### 1.2 Hyperparameter tuning

All `GridSearchCV` operates on the training partition alone using
5-fold stratified inner CV. The winner is chosen by validation ROC-AUC
(a single, held-out evaluation) — not by test performance.

### 1.3 Threshold selection

Every threshold-related computation runs on validation predictions.
The chosen threshold τ = 0.21 is applied unchanged on the test set.
Rerunning threshold optimisation on test predictions would be a
textbook leakage error; the pipeline is structured so this is not
even possible without editing code.

### 1.4 Calibration fitting

Platt scaling is fit on validation-set predictions (Stage 8), never on
train (overfit) and never on test (leakage). This differs from the
common but incorrect pattern of using `CalibratedClassifierCV` with
`cv=5` on training data alone.

### 1.5 SHAP background

The SHAP LinearExplainer uses the **training set** as its background
distribution; SHAP values are computed on the **test set**. This gives
attributions that reflect model behaviour on unseen data, without
letting the test set influence the explainer itself.

### 1.6 Test set access

The 82 test cases are read exactly twice: once for probability
prediction (Stage 6), once for confusion-matrix and bootstrap
computation. There is no loop that consults test performance and
modifies anything upstream.

## 2. Split Design

- **Ratio**: 70 / 15 / 15. Defensible on n = 541: yields 377 / 82 / 82
  cases, with 123 / 27 / 27 PCOS positives respectively. Enough
  positives per split to make Recall estimates non-degenerate.
- **Stratification**: on the binary `PCOS (Y/N)` label, so prevalence
  is preserved across splits.
- **Serialisation**: index lists are serialised to
  `data/splits/splits.json`. Every downstream stage loads these
  indices rather than re-splitting. This is what makes the pipeline
  bit-identical run-to-run.
- **Alternative rejected**: 80 / 10 / 10, which is more common in
  large-data literature. On n = 541 it would leave val and test with
  only 54 cases each (~17 PCOS positives), making Recall estimates
  swing wildly on single mispredictions. Rejected as noisier for
  small-cohort benchmarking.

## 3. Model Selection Protocol

1. Four candidate classifiers with published hyperparameter grids:
   - Logistic Regression (L2 penalty; C ∈ {0.01, 0.1, 1, 10})
   - SVM RBF (C ∈ {0.1, 1, 10, 100}; γ ∈ {scale, auto, 0.01, 0.001})
   - KNN (k ∈ {3, 5, 7, 11, 15}; weights ∈ {uniform, distance})
   - Gaussian NB (var_smoothing ∈ {1e-9, 1e-8, 1e-7, 1e-6})
2. 5-fold stratified `GridSearchCV`, scoring = ROC-AUC, on train only.
3. Refit each winner and score on **validation**.
4. Highest validation ROC-AUC becomes primary reference model.
5. DeLong pairwise test on **test** to check whether the winner is
   statistically better than the alternatives. Result: it is not —
   all pairwise p-values > 0.05 (see [`evaluation_section.md`](evaluation_section.md) § 2).
   This turns "we picked LR" from a performance claim into an
   Occam-style claim, defensible even if a future extension finds
   marginal improvements with a more complex model.

## 4. Metric Suite Rationale

| Metric | Why included |
|--------|--------------|
| **ROC-AUC** | Standard baseline; threshold-independent; robust to class imbalance in ranking terms. |
| **PR-AUC** | Explicit for the positive class; more informative than ROC-AUC on 33 % prevalence. |
| **Brier score** | Reflects both discrimination and calibration; unaffected by threshold. |
| **Recall** | Primary clinical metric for a screening task. |
| **Specificity** | Complements recall; measures the false-alarm cost. |
| **Precision (PPV)** | Post-test probability of PCOS given a positive prediction — the number a clinician actually uses. |
| **F1** | Harmonic mean of precision and recall; sanity check against extreme threshold choices. |
| **Balanced accuracy** | Insulates against class-imbalance illusions. |
| **MCC** | Symmetric across all four confusion-matrix cells; single-number sanity check. |

Accuracy is **omitted** — it is misleading at 33 % prevalence and
would let a "predict-always-negative" classifier score 67 %.

## 5. Statistical Testing

- **DeLong test** (Sun & Xu 2014 fast implementation) for pairwise
  ROC-AUC comparison on the test set. All six pairs among four
  classifiers are reported.
- **Bootstrap** (n = 1000, percentile method, seed = 42) for 95 % CIs
  on every metric. See [ADR-005](adr/ADR-005-bootstrap-confidence-intervals.md).
- **McNemar test** is deliberately omitted for time and because it
  answers a different question (marginal disagreement on predictions,
  not AUC difference).

## 6. Explainability

- **LinearExplainer** (exact, deterministic) for reasons documented in
  [ADR-003](adr/ADR-003-shap-linear-explainer.md).
- **Global**: mean |SHAP| ranking, bar chart, and beeswarm plot on the
  test set.
- **Local**: waterfall plots for one confident True Positive and one
  confident True Negative — chosen automatically as the highest and
  lowest predicted probabilities among correctly classified test
  cases.
- **Clinical grounding**: the top-15 features are mapped to the
  Rotterdam ESHRE/ASRM criteria and the Teede et al. 2018 guideline in
  `results/narratives/shap_interpretation.md`. Fourteen of fifteen
  map to established biomarkers; the exception (Marriage Status) is
  flagged as a probable proxy variable for future ablation.

## 7. Calibration

Two calibration quantities are reported:

- **Brier score** (mean squared error between probability and label).
- **Expected Calibration Error** (ECE) with 10 quantile-binned bins.

Platt (sigmoid) scaling is applied conditionally on raw ECE > 0.05 —
see [ADR-004](adr/ADR-004-platt-scaling-conditional.md). Fit on
validation-set predictions to avoid both train-side over-confidence
and test-side leakage.

## 8. Reproducibility Enforcement

- Single fixed seed (42) at every entry point that touches randomness:
  data split, CV split, model random state, bootstrap RNG.
- Split indices serialised (`data/splits/splits.json`) rather than
  regenerated.
- Pinned dependency versions in `requirements.txt`.
- Warnings suppressed in scripts to keep console output actionable;
  no numerical operation is affected.
- The full pipeline runs in under 20 seconds on a single CPU core —
  cheap enough that reviewers actually rerun it.

## Related Documents

- [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) — formal spec
- [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md) — output artefact contract
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — practical guide
- [`LIMITATIONS.md`](LIMITATIONS.md) — what this methodology does not solve
- [`adr/`](adr/) — decision records
