# Dataset Card — HormoBench-PCOS

> Follows the Hugging Face dataset-card conventions (Gebru et al., 2018).

## Overview

- **Source**: [Polycystic Ovary Syndrome (PCOS)](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos)
  by Prasoon Kottarathil (Kaggle).
- **Original license**: CC BY-NC-SA 4.0.
- **Cleaning license**: this cleaned derivative is redistributed under the
  same CC BY-NC-SA 4.0 license, with attribution preserved.
- **Version used**: latest as of 2026-07 (uploader has not versioned).
- **Cohort**: 541 patients from **10 hospitals across Kerala, India**.
- **Task**: binary classification — presence of PCOS from tabular clinical
  and hormonal features.

## Composition

- **Rows**: 541
- **Features**: 41 (39 continuous, 2 binary encoded as {0, 1})
- **Target**: `PCOS (Y/N)` — binary (1 = PCOS, 0 = no PCOS)
- **Prevalence**: 32.7 % positive (177 / 541)

### Feature groups

| Group | Examples | Count |
|-------|----------|:-----:|
| Demographics | Age, Marriage Status | 3 |
| Anthropometry | Weight, Height, BMI, Waist:Hip Ratio | 6 |
| Reproductive history | Cycle regularity, cycle length, pregnancies | 4 |
| Hormones | LH, FSH, TSH, PRL, AMH, PRG, β-hCG (I & II) | 8 |
| Metabolic | Fasting glucose, Hb, RBS | 3 |
| Ultrasound | Follicle counts (R/L), avg follicle size, endometrium | 5 |
| Symptoms | Hair growth, skin darkening, acne, hair loss, weight gain | 5 |
| Vitals | Systolic BP, Diastolic BP, pulse rate, respiratory rate | 4 |
| Lifestyle | Fast-food exposure, regular exercise | 2 |
| Blood group | Coded | 1 |

## Splits

Provided in `data/splits/splits.json` — stratified by target with seed 42:

| Split | n   | PCOS+ | PCOS+ % |
|-------|:---:|:-----:|:-------:|
| Train | 377 |  123  |  32.6 % |
| Val   |  82 |   27  |  32.9 % |
| Test  |  82 |   27  |  32.9 % |

## Preprocessing Applied to Produce `pcos_clean.csv`

Deterministic, no target-aware operations:

1. Column names: stripped of leading/trailing whitespace; internal
   double spaces collapsed.
2. Dropped columns: `Unnamed: 44` (junk; 539/541 nulls), `Sl. No` and
   `Patient File No.` (patient identifiers, not features).
3. Fixed two isolated typos: `'1.99.'` → `1.99` (β-hCG II);
   `'a'` → `NaN` (AMH).
4. All remaining columns coerced to numeric. Three NaN cells retained
   (0.18 % of the matrix) for in-pipeline median imputation.

Imputation and scaling are intentionally **not** applied at this stage —
they are fit inside training folds. See [`METHODOLOGY.md`](METHODOLOGY.md)
§ 1.1.

## Uses

### Intended

- Benchmarking classical and modern classifiers on tabular clinical
  data for PCOS risk stratification.
- Teaching medical-AI methodology (anti-leakage, calibration, SHAP).
- Ablation and extension studies (adding features, changing splits,
  adding metrics) against a fixed baseline.

### Out of scope

- **Clinical deployment** — this dataset is neither representative
  enough nor labelled reliably enough to underpin diagnostic tooling.
- **Population studies** — the cohort is drawn from a specific
  geographic region and healthcare-seeking population.
- **Longitudinal analysis** — the dataset is cross-sectional.

## Known Biases and Limitations

- **Geographic bias**: 10 hospitals in Kerala, India. Findings may not
  transfer to European, African, East Asian, or American cohorts,
  whose phenotypic presentation and hormonal norms differ.
- **Selection bias**: patients presenting to fertility clinics or
  general hospitals with symptoms — not a general-population sample.
- **Label uncertainty**: PCOS labels are based on the treating team's
  application of Rotterdam-era criteria. Inter-rater reliability is
  unknown.
- **Feature coding ambiguity**: some binary features (`Y/N`) do not
  distinguish "never" from "not asked". A `NaN` in `Fast food (Y/N)`
  probably means "not recorded", but this is not documented in the
  source.
- **Sample size**: 541 patients is small for robust generalisation
  claims. Confidence intervals in results reflect this.
- **Class imbalance**: 32.7 % positive is mild imbalance but pushes
  metric choice toward Recall and PR-AUC over Accuracy.

## Ethical Considerations

- The data are already de-identified by the original uploader.
- No re-identification attempts are performed.
- HormoBench-PCOS explicitly renounces diagnostic claims — see
  [`LIMITATIONS.md`](LIMITATIONS.md).
- The cohort represents a specific socio-economic and geographic
  slice; deploying a model trained on this dataset in a different
  context without recalibration would be irresponsible.

## Citation of the Source Dataset

If you use this benchmark you must cite the original data uploader:

> Kottarathil, P. Polycystic Ovary Syndrome (PCOS) [Data set]. Kaggle.
> https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos

## Second Cohort: Follicular-Fluid Amino-Acid Metabolomics Track

A separate, smaller cohort is benchmarked as its own track. It is **not**
merged with the Kerala data: the two share only Age, Height, Weight and BMI.

| Item | Value |
|---|---|
| Raw files | `data/raw/metabolomics/Table-S1.xlsx` (clinical), `Table-S2.xlsx` (amino acids) |
| Clean file | `data/processed/pcos_metabolomics_clean.csv` |
| Size | 72 participants: 37 controls, 35 PCOS (48.6 %) |
| Features | Age, Height (m), Weight (kg), BMI, serum testosterone (ng/ml), 17 follicular-fluid amino acids (LC-MS peak areas) |
| Split | 80/20 stratified, seed 42 → `data/splits/metabolomics_splits.json` (57 train / 15 test) |
| Scripts | `scripts/clean_metabolomics.py`, `scripts/train_metabolomics.py` |
| Results | `results/metabolomics/results.json`, `results/metabolomics/test_metrics.md` |

**Removed during cleaning**

- *Estrous cyclicity* and *Ovarian morphology*: Rotterdam diagnostic criteria
  that separate the two groups perfectly (every control is Regular/Normal,
  every PCOS case is irregular/polycystic). Keeping them leaks the label.
- *L-Glycine*, *L-Alanine*, *L-Cysteine*: marked "/" (not measured) for every participant.

**Caveats**

- With 15 test participants, test-set CIs span roughly ±0.25 AUC. Treat the
  5-fold CV AUC on the training set as the more stable signal.
- Serum testosterone reflects biochemical hyperandrogenism, which is also a
  Rotterdam criterion; the `amino_acids` feature set excludes it.
- Amino-acid values are raw instrument peak areas, not absolute concentrations.
- Amino acids come from an untargeted metabolomic survey of **follicular
  fluid** (collected during IVF/oocyte retrieval), not blood. A model built
  on them is not a screening tool for the general population.

**Source and licence**: CC BY 4.0. Cite as:

> Yuan, Y. (2023). *DataSet for PCOS* (Version 1) [Data set]. Mendeley Data.
> https://doi.org/10.17632/mh94mxn3nh.1

## Related Documents

- [`MODEL_CARD.md`](MODEL_CARD.md) — the model trained on this dataset
- [`LIMITATIONS.md`](LIMITATIONS.md) — bounds on what can be claimed
- [`METHODOLOGY.md`](METHODOLOGY.md) — how the data are used
