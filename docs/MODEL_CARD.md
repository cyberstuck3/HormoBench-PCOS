# Model Card — HormoBench-PCOS Reference Model

> Follows the model-card conventions of Mitchell et al. (2019).

## Model Details

- **Name**: HormoBench-PCOS Reference Model v0.1.0
- **Architecture**: L2-regularised Logistic Regression
- **Hyperparameters**: C = 0.01, penalty = L2, max_iter = 1000, seed = 42
- **Preprocessing**: median imputation + standard scaling, fit on train only
- **Calibration**: Platt (sigmoid) scaling, fit on validation-set predictions
- **Decision threshold**: τ = 0.21 (selected on validation, applied to test)
- **Framework**: scikit-learn 1.8.0
- **Training runtime**: <5 seconds on a single CPU core
- **File**: `results/models/best_model.joblib` (unfrozen);
  Platt-scaled probabilities are computed on the fly in `run_rigor.py`

## Intended Use

- **Primary**: reference baseline for PCOS risk prediction benchmarks;
  starting point for methodological research on tabular clinical AI.
- **Secondary**: teaching example for anti-leakage architecture,
  calibration analysis, and SHAP-based interpretation.

## Out-of-Scope Use

- **Clinical decision support without further validation**. This model
  was trained on 377 patients from one geographic region and has not
  undergone prospective clinical validation, regulatory review, or
  workflow integration testing.
- **Deployment on populations that differ substantially from the
  training cohort** (Kerala hospital patients) without recalibration
  and revalidation. Hormonal reference ranges and PCOS phenotype
  prevalence vary considerably by ancestry and healthcare access.
- **Screening in children or peri-menopausal women** — the training
  data cover reproductive-age adults only.
- **Standalone diagnosis**. PCOS diagnosis requires clinical application
  of the Rotterdam criteria; a probability score is not a substitute
  for that process.

## Factors

- **Cohort**: adult women, primarily reproductive age, presenting to
  10 hospitals across Kerala, India.
- **Environment**: outpatient clinical setting with access to hormone
  panels and pelvic ultrasound.

## Metrics (Held-Out Test Set, n = 82)

Threshold-independent:

| Metric | Point | 95 % CI |
|--------|:-----:|:-------:|
| ROC-AUC | 0.951 | [0.878, 0.995] |
| PR-AUC | 0.932 | [0.852, 0.989] |
| Brier score (raw) | 0.095 | [0.069, 0.126] |
| Brier score (Platt) | 0.076 | — |

At the tuned threshold τ = 0.21:

| Metric | Point | 95 % CI |
|--------|:-----:|:-------:|
| Recall (Sensitivity) | 0.963 | [0.867, 1.000] |
| Specificity | 0.636 | [0.510, 0.764] |
| Precision (PPV) | 0.565 | [0.419, 0.705] |
| F1 | 0.712 | [0.580, 0.822] |
| Balanced accuracy | 0.800 | [0.723, 0.871] |
| Matthews CC | 0.568 | [0.419, 0.711] |

**Baseline comparisons** (5-fold stratified CV, mean ± SD):

| Model | ROC-AUC | PR-AUC |
|-------|:-------:|:------:|
| Logistic Regression (this) | 0.952 ± 0.016 | 0.922 ± 0.017 |
| SVM (RBF) | 0.948 ± 0.015 | 0.914 ± 0.018 |
| KNN | 0.933 ± 0.029 | 0.901 ± 0.026 |
| Gaussian NB | 0.908 ± 0.015 | 0.822 ± 0.048 |

**DeLong test** on the test set: LR is not significantly better than
the other three models (all p > 0.05). The choice of LR is a
transparency-first, not performance-first, decision. See
[ADR-001](adr/ADR-001-logistic-regression-baseline.md).

## Calibration

Raw ECE was 0.159 (poor) — the L2 regularisation makes the raw model
under-confident. After Platt scaling on validation predictions, ECE
falls to 0.080 and Brier from 0.095 to 0.076. Calibrated probabilities
are recommended for downstream use. See
[ADR-004](adr/ADR-004-platt-scaling-conditional.md) and the
reliability diagram in `results/figures/calibration.png`.

## Explainability

Global feature importance (top 5, mean |SHAP|):

1. LH (Luteinizing hormone) — 0.66
2. Follicle count, right ovary — 0.30
3. Follicle count, left ovary — 0.27
4. Weight gain — 0.23
5. Hair growth (hirsutism) — 0.22

**Fourteen of the fifteen top features** map to established Rotterdam
criteria or their modern refinements (Teede et al., 2018). The
literature-linked table is in
`results/narratives/shap_interpretation.md`.

## Ethical Considerations

- Trained on a single-region cohort; performance on other populations
  is unknown and may be worse.
- Overconfident deployment could lead to over-referral or
  under-referral, both harmful.
- The model is intentionally interpretable so clinicians can inspect
  and challenge individual predictions.
- Fairness across demographic subgroups (age band, BMI category,
  socioeconomic proxies) has not been evaluated; a stratified
  fairness audit is a stated future-work item in
  [`LIMITATIONS.md`](LIMITATIONS.md).

## Caveats and Recommendations

- **Do not use raw probabilities** unless the Platt calibration is
  applied — the raw scores are systematically low.
- **Do not shift the decision threshold** without repeating the
  validation-set threshold analysis.
- **Do report confidence intervals** alongside any evaluation on
  external data.
- **Do keep the reference model versioned** — a change to the
  hyperparameters requires a version bump and a re-run of the
  DeLong test.

## Related Documents

- [`DATASET_CARD.md`](DATASET_CARD.md)
- [`LIMITATIONS.md`](LIMITATIONS.md)
- [`evaluation_section.md`](evaluation_section.md) — full results
