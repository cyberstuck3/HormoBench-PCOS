# Benchmark Specification

> **Purpose of this document.** Define HormoBench-PCOS as a formal benchmark,
> independent of the Python reference implementation. A researcher using R,
> Julia, or any other language should be able to build a conforming
> implementation from this specification alone.

## 1. Scope

HormoBench-PCOS is a **binary classification benchmark** for PCOS risk
prediction from tabular clinical records. It provides:

- A **fixed cleaned dataset** (541 patients, 42 columns after preprocessing)
- A **fixed 70/15/15 stratified split** (seed = 42)
- A **reference model** (Logistic Regression, L2, C = 0.01)
- A **fixed evaluation protocol** covering nine metrics with bootstrap
  95% confidence intervals
- A **calibration and interpretability contract** covering ECE, Brier,
  DeLong pairwise tests, and SHAP feature attributions

## 2. Inputs

### 2.1 Dataset schema

The source is the cleaned dataset at `data/processed/pcos_clean.csv`,
derived deterministically from the Kaggle "Polycystic Ovary Syndrome"
dataset (Prasoon Kottarathil, 10-hospital Kerala cohort). See
[`DATASET_CARD.md`](DATASET_CARD.md) for provenance and biases.

- **Rows**: 541
- **Columns**: 42 (41 features + 1 binary target `PCOS (Y/N)`)
- **Types**: all numeric after cleaning; three NaN cells retained for
  in-pipeline imputation
- **Target prevalence**: 32.7% (177 positive / 364 negative)

### 2.2 Configuration parameters

| Parameter | Value | Rationale |
|-----------|:-----:|-----------|
| `SEED` | 42 | Standard reproducibility seed |
| `SPLIT` | 70 / 15 / 15 stratified | Defensible on n = 541; see [ADR-005](adr/) |
| `DECISION_THRESHOLD` | 0.26 | Selected on val to maximise recall s.t. precision ≥ 0.65 |
| `N_BOOTSTRAP` | 1000 | Standard for 95% CI stability |
| `CV_FOLDS` | 5 | Stratified k-fold on training set |
| `ECE_THRESHOLD_FOR_PLATT` | 0.05 | Apply Platt scaling only if raw ECE exceeds this |

## 3. Pipeline Stages

Every conforming implementation must execute stages in the following order.
Stages are pure functions of their inputs; no stage may consume information
from a later stage's data.

### Stage 1 — Load & Clean
- Read raw xlsx (or the cleaned CSV directly)
- Strip whitespace in column names, collapse internal double spaces
- Drop identifier columns and one junk column (`Unnamed: 44`)
- Coerce two typo entries (`'1.99.'`, `'a'`) to numeric / NaN
- **Output**: 541 × 42 numeric DataFrame with 3 NaN cells

### Stage 2 — Split
- Stratified 70/15/15 by `PCOS (Y/N)` with `random_state = SEED`
- Serialise index lists as JSON — every downstream stage must use these
  exact indices

### Stage 3 — Preprocess (fit on train only)
- Median imputation (fit on train, applied to val and test)
- Standard scaling (fit on train, applied to val and test)

### Stage 4 — Train + Tune
- 5-fold stratified CV `GridSearchCV`, scoring = ROC-AUC
- Full preprocessing pipeline refit inside every fold (no leakage)
- Four candidate classifiers: LR (L2), SVM (RBF), KNN, GaussianNB
- Reported hyperparameter grids in [ADR-001](adr/) and code

### Stage 5 — Threshold Optimisation
- Evaluated on the **validation set only**
- Objective: maximise recall subject to precision ≥ 0.65
- The chosen threshold is applied unchanged on the test set

### Stage 6 — Evaluate (test set, single pass)
- All threshold-independent metrics (ROC-AUC, PR-AUC, Brier)
- All threshold-dependent metrics at the chosen τ
- Non-parametric bootstrap (n = 1000) for 95% CIs

### Stage 7 — Compare (DeLong pairwise)
- Pairwise ROC-AUC comparison across all four models on the test set
- Sun & Xu (2014) fast implementation

### Stage 8 — Calibrate
- Compute Brier score and Expected Calibration Error (10 quantile bins)
- **Conditional**: fit Platt (sigmoid) scaling on validation predictions
  if raw ECE > 0.05; otherwise report only raw calibration

### Stage 9 — Explain
- SHAP LinearExplainer on the winning LR model
- Background = training set; explanations computed on test set
- Global (mean |SHAP|) + local (one confident TP, one confident TN)

### Stage 10 — Export
- Machine-readable JSON dump of every number
- Markdown tables for direct README / paper embedding
- Publication-quality figures at ≥ 150 DPI
- Consolidated paper-style write-up at `docs/evaluation_section.md`

## 4. Outputs

See [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md) for the artefact-level contract.

## 5. Reproduction Contract

Given:
- The same source data at `data/raw/PCOS_data_without_infertility.xlsx`
- The same commit hash on `main`
- The pinned Python and library versions in `requirements.txt`
- The same operating system family (Linux x86-64 was used for the
  reference run)

Then `python scripts/benchmark.py` must produce identical values in
`results/rigor_all_results.json` up to floating-point rounding at
1e-9 tolerance.

## 6. Metrics — Formal Definitions

Let TP, FP, TN, FN denote confusion-matrix cells at threshold τ.

- **Recall (Sensitivity)** = TP / (TP + FN)
- **Specificity** = TN / (TN + FP)
- **Precision (PPV)** = TP / (TP + FP)
- **F1** = 2 · Precision · Recall / (Precision + Recall)
- **Balanced Accuracy** = (Recall + Specificity) / 2
- **Matthews CC** = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
- **ROC-AUC** = area under the ROC curve (threshold-independent)
- **PR-AUC** = average precision (threshold-independent)
- **Brier score** = mean squared error between predicted probability and label
- **ECE** = ∑ᵢ (|Bᵢ| / n) · |acc(Bᵢ) − conf(Bᵢ)|, quantile-binned

## 7. Extension Points

A researcher can extend HormoBench-PCOS without modifying the specification:

- **New model**: add an entry to `MODEL_GRIDS` in `scripts/train_baseline.py`;
  it will automatically be compared via DeLong and CV.
- **New metric**: add to `src/hormobench/rigor/metrics.py` under
  `PROB_METRICS` or `PRED_METRICS`.
- **New dataset**: replace `data/raw/*` and rerun; the pipeline is
  schema-agnostic apart from the `PCOS (Y/N)` target column name.
- **New explainer**: replace `scripts/explain.py`; downstream consumers
  read only `results/shap_importance.csv`.

## Related Documents

- [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md) — output artefact contract
- [`METHODOLOGY.md`](METHODOLOGY.md) — technical justification of each stage
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — practical reproduction guide
