# Limitations

> **Purpose of this document.** State — as plainly as possible — what
> HormoBench-PCOS does not, cannot, and should not do. The Hack-Nation
> challenge briefing explicitly identifies "unsupported medical or
> diagnostic claims" as a mark of a weak submission. This document is
> the direct response.

## 1. Not a Diagnostic Tool

HormoBench-PCOS produces a probabilistic risk score. It does not
diagnose PCOS. Diagnosis requires clinical evaluation applying the
Rotterdam criteria, physician judgement, and confirmatory testing.
The model exists to support research, not clinical decision-making.

## 2. Cohort and Generalisation

The training data are from **10 hospitals in Kerala, India**, with
541 patients total. Consequences:

- **Geographic transfer is unproven.** Hormonal reference ranges,
  BMI distribution, and PCOS phenotype prevalence vary across
  populations. A model tuned on this cohort may misperform on
  European, African, East Asian, or American cohorts.
- **Selection bias is present.** Patients presenting to fertility or
  general hospitals with symptoms are not representative of women in
  the general population — most of whom never present.
- **Age range is limited.** The dataset covers reproductive-age
  adults. Adolescents and peri-menopausal women are not represented,
  yet PCOS presentation differs substantially in both groups.

## 3. Label Quality

- **Rotterdam criteria vary in application.** Different clinicians
  weight the three Rotterdam criteria (anovulation, hyperandrogenism,
  polycystic ovarian morphology) differently. The dataset's labels
  reflect the specific application by the source hospitals.
- **Inter-rater reliability unknown.** The source dataset does not
  report kappa or double-rating.
- **No follow-up validation.** Some "PCOS negative" labels may
  reflect early or subclinical disease that would be diagnosed later.

## 4. Feature Coding Ambiguities

- Binary `Y/N` features do not distinguish "no" from "not asked" from
  "not recorded". A `NaN` in `Fast food (Y/N)` is treated as missing
  and imputed with the training median — but the true distinction is
  clinically meaningful and unrecoverable.
- Some measurements (e.g. β-hCG II) have wide dynamic ranges and
  probable outliers that reflect data-entry errors rather than
  biology. The pipeline does not currently trim these.

## 5. Sample Size

At n = 541 total, and only 82 test cases with 27 PCOS positives:

- **Confidence intervals are wide.** ROC-AUC 95 % CI spans 0.878 to
  0.995 — the point estimate is not the same as the true
  performance.
- **Sub-group analyses are not statistically powered.** Fairness
  audits across age bands or BMI categories would leave 10–20
  patients per cell — too few to draw conclusions.
- **DeLong tests are conservative.** All pairwise comparisons show
  p > 0.05, but that is compatible with real differences of a few
  percentage points that the sample cannot detect.

## 6. What the Model Learns vs What It Diagnoses

The SHAP analysis shows that fourteen of fifteen top features map to
established Rotterdam criteria — evidence that the model has learned
clinically meaningful signal. But: correlation with the Rotterdam
criteria and *being* a Rotterdam diagnosis are different things.

- The **rank-1 feature is LH**, whose SHAP magnitude is more than
  twice any other. This reflects the training distribution: high LH
  is strongly predictive here, but LH:FSH ratio inversion is
  neither a Rotterdam criterion nor uniformly present in PCOS.
- The **Marriage Status feature** appears in the top-15. This is
  almost certainly a proxy for age or fertility-seeking behaviour
  in this cohort, not a causal PCOS factor. A future ablation study
  should remove it.

## 7. Methodological Choices Not Made

Explicit non-decisions, listed so extenders know where to look:

- **No feature engineering.** Ratios (LH:FSH, waist:hip already
  present), interactions, and non-linear transforms are not added.
- **No missing-data model.** Simple median imputation. Multiple
  imputation would be more principled.
- **No outlier handling.** Extreme β-hCG values pass through unchanged.
- **No sub-group calibration.** Calibration is evaluated globally,
  not per age or BMI band.
- **No non-linear models (trees, GBMs, neural nets).** By design;
  see [ADR-001](adr/ADR-001-logistic-regression-baseline.md).
- **No temporal or longitudinal modelling.** The dataset is
  cross-sectional; a hormone snapshot is a poor proxy for a
  time-varying biological state.

## 8. Reproducibility Boundaries

- The pipeline is bit-identical *given* the same commit, seed,
  Python version, and library versions in `requirements.txt`.
- Different **matplotlib backends** may render figures with minor
  pixel-level differences; the numeric content is unaffected.
- Different **joblib versions** may serialise the model differently;
  the model itself is unchanged.
- **Alternative operating systems** were not tested. The reference
  run is Linux x86-64.

## 9. What Would Fix Each of These

| Limitation | Concrete extension that would remedy it |
|------------|-----------------------------------------|
| Geographic bias | Add cohorts from PhysioNet mcPHASES + NHANES, refit, recalibrate |
| Small sample | Combine multiple public PCOS cohorts into a meta-benchmark |
| Label uncertainty | Add inter-rater–reviewed re-labelling on a validation subset |
| Feature ambiguity | Enforce a data dictionary with explicit "not asked" codes |
| Fairness audits | Larger cohort → per-subgroup metric reporting |
| Longitudinal modelling | Move to a temporal dataset (mcPHASES has this) |

## 10. Non-Goals

HormoBench-PCOS deliberately does **not** aim to:

- Beat the state-of-the-art on this dataset.
- Deploy as a clinical product.
- Replace clinical judgement.
- Substitute for the Rotterdam workup.
- Serve as evidence of population-level PCOS prevalence.

## Related Documents

- [`PROJECT_PHILOSOPHY.md`](PROJECT_PHILOSOPHY.md) — why these
  limitations are acceptable given the project's goals
- [`DATASET_CARD.md`](DATASET_CARD.md) — data provenance
- [`MODEL_CARD.md`](MODEL_CARD.md) — model-level caveats
