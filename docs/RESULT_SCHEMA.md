# Result Schema

> **Purpose of this document.** Define the contract of every artefact produced
> by `scripts/benchmark.py`. Consumers can rely on these paths, formats, and
> schemas remaining stable within a major version.

## Directory Layout After a Successful Run

```
results/
├── rigor_all_results.json          ← primary machine-readable dump
├── results.json                    ← baseline (Stage 2) hyperparameter results
├── shap_importance.csv             ← ranked feature importances
├── figures/
│   ├── roc_curves.png              ← 4-model ROC (Fig for README §5)
│   ├── pr_curves.png               ← 4-model PR with prevalence baseline
│   ├── calibration.png             ← reliability diagram, raw + Platt
│   ├── cv_boxplots.png             ← 5-fold CV distributions per model
│   ├── shap_global.png             ← bar + beeswarm, top 15 features
│   ├── shap_waterfall_TP.png       ← local explanation, confident TP
│   └── shap_waterfall_TN.png       ← local explanation, confident TN
├── tables/
│   ├── test_metrics.md             ← headline metrics with 95% CI
│   ├── cv_comparison.md            ← per-model mean ± SD
│   ├── delong_pairwise.md          ← pairwise ΔAUC + p-values
│   └── calibration.md              ← Brier + ECE, raw and Platt
├── narratives/
│   ├── threshold_justification.md  ← clinical rationale paragraph
│   └── shap_interpretation.md      ← literature-linked feature narrative
└── models/
    ├── best_model.joblib           ← trained LR
    └── preprocessor.joblib         ← fitted imputer + scaler pipeline

docs/
└── evaluation_section.md           ← paper-style consolidation of above
```

## Per-Artefact Schemas

### `results/rigor_all_results.json`

Top-level JSON object:

```json
{
  "config": {
    "seed":         42,
    "threshold":    0.21,
    "n_bootstrap":  1000
  },
  "test_metrics": {
    "<metric_name>": {
      "point":   float,
      "lo":      float,   // 2.5th percentile
      "hi":      float,   // 97.5th percentile
      "std":     float,
      "n_valid": int      // successful bootstrap iterations
    }
  },
  "cv_results": [
    {
      "model":    str,
      "metric":   "roc_auc" | "average_precision",
      "mean":     float,
      "std":      float,
      "per_fold": [float, float, float, float, float]
    }
  ],
  "delong": {
    "aucs":     { "<model_name>": float },
    "pairwise": {
      "<model_a> vs <model_b>": {
        "auc_a":                float,
        "auc_b":                float,
        "delta":                float,
        "se":                   float,
        "z":                    float,
        "p_value":              float,
        "significant_at_0.05":  bool
      }
    }
  },
  "calibration": {
    "raw":           { "brier": float, "ece": float,
                       "prob_true": [...], "prob_pred": [...] },
    "platt_applied": bool,
    "after_platt":   null | { "brier": float, "ece": float,
                              "prob_true": [...], "prob_pred": [...] }
  },
  "confusion_matrices": {
    "default_0.5": { "fn": int, "fp": int },
    "tuned_0.21":  { "fn": int, "fp": int }
  }
}
```

Metric names present under `test_metrics`:
`roc_auc`, `pr_auc`, `brier`, `precision`, `recall`, `specificity`, `f1`,
`balanced_accuracy`, `mcc`.

### `results/results.json` (Stage 2 baseline)

```json
{
  "best_model":  str,
  "best_params": { ... },
  "all_models":  {
    "<model_name>": {
      "best_params":  { ... },
      "cv_auc_mean":  float,
      "val_auc":      float,
      "val_f1":       float
    }
  }
}
```

### `results/shap_importance.csv`

Two columns, sorted by `mean_abs_shap` descending:

| column          | type   | description                            |
|-----------------|--------|----------------------------------------|
| `feature`       | string | feature name from cleaned CSV          |
| `mean_abs_shap` | float  | mean absolute SHAP value on test set   |

### `results/figures/*.png`

All figures are PNG at 150 DPI, sizes designed for embedding in a paper
or README at 100% width without further scaling.

| File                       | Size (in)    | Consumer  |
|----------------------------|--------------|-----------|
| `roc_curves.png`           | 7 × 6        | README §5 |
| `pr_curves.png`            | 7 × 6        | README §5 |
| `calibration.png`          | 7 × 6        | README §4 |
| `cv_boxplots.png`          | 13 × 5       | README §3 or docs |
| `shap_global.png`          | 18 × 8       | README §5, docs   |
| `shap_waterfall_TP.png`    | 10 × 8       | docs      |
| `shap_waterfall_TN.png`    | 10 × 8       | docs      |

### `results/tables/*.md`

Markdown tables using standard GitHub-flavoured syntax. Directly embeddable
in README or paper without conversion.

### `results/models/*.joblib`

- `best_model.joblib` — fitted sklearn estimator (LogisticRegression with
  the tuned hyperparameters).
- `preprocessor.joblib` — fitted `Pipeline([SimpleImputer, StandardScaler])`.

Load with `joblib.load(path)`. Requires the pinned scikit-learn version to
guarantee round-trip fidelity.

### `docs/evaluation_section.md`

Auto-generated paper-style write-up. Composed by Stage 4 from the tables,
narratives, and per-model hyperparameter dictionaries. Overwritten on every
benchmark run — do not edit by hand; edit the generator in
`scripts/run_rigor.py` instead.

## Determinism Guarantees

| Artefact                          | Bit-identical | Numerically identical |
|-----------------------------------|:-------------:|:---------------------:|
| `data/processed/pcos_clean.csv`   | ✅            | ✅                    |
| `data/splits/splits.json`         | ✅            | ✅                    |
| `results/rigor_all_results.json`  | —             | ✅ (to 1e-9)          |
| `results/tables/*.md`             | ✅            | ✅                    |
| `results/models/*.joblib`         | ⚠ platform-dep| ✅                    |
| `results/figures/*.png`           | ⚠ renderer-dep| N/A                   |

Bit-differences in `.png` and `.joblib` files are expected across
matplotlib backends and joblib versions; the numeric content they encode
is deterministic.

## Consumer Guide

```python
# Load the primary results
import json
with open("results/rigor_all_results.json") as f:
    r = json.load(f)

# Headline test-set numbers
r["test_metrics"]["roc_auc"]["point"]        # → 0.951
r["test_metrics"]["roc_auc"]["lo"]           # → 0.878 (2.5%)
r["test_metrics"]["roc_auc"]["hi"]           # → 0.995 (97.5%)

# Load the trained pipeline for inference
import joblib, pandas as pd
prep = joblib.load("results/models/preprocessor.joblib")
mdl  = joblib.load("results/models/best_model.joblib")
X    = pd.read_csv("data/processed/pcos_clean.csv").drop(columns=["PCOS (Y/N)"])
prob = mdl.predict_proba(prep.transform(X))[:, 1]
```

## Backward Compatibility

Schema changes bump the minor version and are logged in `CHANGELOG.md`
(to be added at first backward-incompatible change). Field additions are
non-breaking; field removals or type changes are breaking.

## Related Documents

- [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) — the pipeline that produces
  these artefacts
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — how to regenerate them
