"""
Stage 4 — Full scientific rigor analysis: bootstrap CIs, PR-AUC, calibration,
DeLong pairwise tests, CV mean ± SD across all four models, plus paper-style
export bundle (tables, narratives, and a consolidated evaluation section).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

from hormobench import (
    CLEAN_CSV, SPLITS_JSON, BASELINE_JSON, RIGOR_JSON, SHAP_CSV,
    MODEL_JOBLIB, PREPROC_JOBLIB, RESULTS, RESULTS_FIG, RESULTS_TAB,
    RESULTS_NAR, DOCS, SEED, N_BOOTSTRAP, load_decision_threshold,
)
from hormobench.registry import MODEL_REGISTRY, build_model
from hormobench.rigor.metrics     import compute_all_metrics
from hormobench.rigor.delong      import delong_pairwise
from hormobench.rigor.calibration import (
    calibration_analysis, apply_platt_scaling, ECE_THRESHOLD_FOR_PLATT,
)
from hormobench.rigor.cv_analysis import cv_compare_models
from hormobench.rigor.figures     import (
    plot_calibration, plot_pr_curve, plot_roc_multi, plot_cv_comparison,
)
from hormobench.rigor.reports     import (
    metrics_table_md, cv_table_md, delong_table_md, calibration_table_md,
)
from hormobench.rigor.clinical    import threshold_justification, shap_interpretation_md

TARGET = "PCOS (Y/N)"


def main():
    for d in [RESULTS_FIG, RESULTS_TAB, RESULTS_NAR]:
        d.mkdir(parents=True, exist_ok=True)

    df      = pd.read_csv(CLEAN_CSV)
    splits  = json.load(open(SPLITS_JSON))
    prior   = json.load(open(BASELINE_JSON))
    best_params = {n: r["best_params"] for n, r in prior["all_models"].items()}
    prep    = joblib.load(PREPROC_JOBLIB)

    # ── Dynamic winner from Stage 2 (C2) ──
    best_name = prior["best_model"]
    # ── Load persisted threshold from Stage 5 (C1) ──
    threshold = load_decision_threshold()

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, y_tr = X.loc[splits["train_idx"]], y.loc[splits["train_idx"]]
    X_va, y_va = X.loc[splits["val_idx"]],   y.loc[splits["val_idx"]]
    X_te, y_te = X.loc[splits["test_idx"]],  y.loc[splits["test_idx"]]

    X_tr_pp, X_va_pp, X_te_pp = prep.transform(X_tr), prep.transform(X_va), prep.transform(X_te)

    # ── Reconstruct + fit all models from shared registry (C3) ──
    models, test_probs = {}, {}
    for name in MODEL_REGISTRY:
        if name not in best_params:
            continue  # skip models not in this run's results
        mdl = build_model(name, best_params[name])
        mdl.fit(X_tr_pp, y_tr)
        models[name]     = mdl
        test_probs[name] = mdl.predict_proba(X_te_pp)[:, 1]

    best_model = models[best_name]
    y_prob     = test_probs[best_name]

    # ── (1) Bootstrap CIs on the test set ──
    print("Computing bootstrap 95% CIs (n=1000) on test set...")
    test_metrics = compute_all_metrics(
        y_te.values, y_prob, threshold=threshold,
        n_iter=N_BOOTSTRAP, seed=SEED,
    )

    # ── (2) 5-fold stratified CV comparison ──
    print("Running 5-fold stratified CV comparison...")
    cv_models = {
        n: build_model(n, best_params[n])
        for n in models
    }
    cv_df = cv_compare_models(cv_models, X_tr, y_tr, seed=SEED)

    # ── (3) DeLong pairwise ──
    print("Computing DeLong pairwise ROC-AUC tests...")
    delong_pw, delong_aucs = delong_pairwise(y_te.values, test_probs)

    # ── (4) Calibration + conditional Platt scaling ──
    print("Analyzing calibration...")
    cal_raw   = calibration_analysis(y_te.values, y_prob)
    cal_after, y_prob_cal, platt = None, None, False
    if cal_raw["ece"] > ECE_THRESHOLD_FOR_PLATT:
        print(f"  ECE={cal_raw['ece']:.4f} > {ECE_THRESHOLD_FOR_PLATT} → applying Platt scaling.")
        y_prob_cal, _ = apply_platt_scaling(best_model, X_va_pp, y_va, X_te_pp)
        cal_after     = calibration_analysis(y_te.values, y_prob_cal)
        platt = True
    else:
        print(f"  ECE={cal_raw['ece']:.4f} ≤ {ECE_THRESHOLD_FOR_PLATT} → raw calibration adequate.")

    # ── (5) Clinical narratives ──
    _, fp_d, fn_d, _ = confusion_matrix(y_te, (y_prob >= 0.50).astype(int)).ravel()
    _, fp_t, fn_t, _ = confusion_matrix(y_te, (y_prob >= threshold).astype(int)).ravel()
    threshold_md = threshold_justification(threshold, fn_d, fp_d, fn_t, fp_t)

    shap_imp = pd.read_csv(SHAP_CSV)
    top15 = list(zip(shap_imp["feature"].head(15), shap_imp["mean_abs_shap"].head(15)))
    shap_md = shap_interpretation_md(top15)

    # ── Figures ──
    print("Rendering figures...")
    plot_calibration(
        y_te.values, y_prob, y_prob_cal=(y_prob_cal if platt else None),
        brier_raw=cal_raw["brier"],
        brier_cal=(cal_after["brier"] if platt else None),
        out_path=RESULTS_FIG / "calibration.png",
    )
    plot_pr_curve(y_te.values,  test_probs, out_path=RESULTS_FIG / "pr_curves.png")
    plot_roc_multi(y_te.values, test_probs, out_path=RESULTS_FIG / "roc_curves.png")
    plot_cv_comparison(cv_df,               out_path=RESULTS_FIG / "cv_boxplots.png")

    # ── Tables + narratives ──
    print("Writing markdown tables and narratives...")
    mtable = metrics_table_md(
        test_metrics,
        title=f"Held-Out Test Set Performance — {best_name} at τ={threshold} "
              f"(95% CI, {N_BOOTSTRAP} bootstrap iterations)",
    )
    ctable = cv_table_md(cv_df)
    dtable = delong_table_md(delong_pw)
    ktable = calibration_table_md(cal_raw, cal_after)

    (RESULTS_TAB / "test_metrics.md").write_text(mtable)
    (RESULTS_TAB / "cv_comparison.md").write_text(ctable)
    (RESULTS_TAB / "delong_pairwise.md").write_text(dtable)
    (RESULTS_TAB / "calibration.md").write_text(ktable)
    (RESULTS_NAR / "threshold_justification.md").write_text(threshold_md)
    (RESULTS_NAR / "shap_interpretation.md").write_text(shap_md)

    # ── Consolidated paper-style section ──
    winner_params = best_params[best_name]
    params_str = ", ".join(f"{k}={v}" for k, v in winner_params.items())
    paper = f"""# HormoBench-PCOS — Evaluation

*Reproducible one-command benchmark. All numbers below were regenerated by
`python scripts/benchmark.py` with seed = {SEED}, decision threshold
τ = {threshold} (selected on validation set), and {N_BOOTSTRAP} bootstrap iterations.*

## 1. Model Selection

Four candidate classifiers — Logistic Regression, SVM (RBF), k-NN, and
Gaussian Naive Bayes — were tuned via 5-fold stratified cross-validation
(`GridSearchCV`, scoring = ROC-AUC) on the training set. Preprocessing
(median imputation + standard scaling) was refit inside every CV fold.

{ctable}

{best_name} ({params_str}) was selected as the reference model based on
validation-set ROC-AUC and Occam's-razor considerations.

## 2. DeLong Pairwise Comparison

{dtable}

## 3. Held-Out Test Performance

Test set: n = {len(y_te)}, {int(y_te.sum())} PCOS positives. Evaluated once.

{mtable}

{threshold_md}

## 4. Calibration

{ktable}

{'Platt scaling improved calibration meaningfully; calibrated probabilities are recommended for downstream clinical use.'
 if platt else
 'The raw model was adequately calibrated; Platt scaling was not applied.'}

## 5. Explainability

{shap_md}

## 6. Reproducibility

- Fixed seed: **{SEED}**  |  Splits: 70/15/15 stratified, saved in `data/splits/splits.json`
- Bootstrap iterations: **{N_BOOTSTRAP}**
- Regenerate with: `python scripts/benchmark.py`
"""
    (DOCS / "evaluation_section.md").write_text(paper)

    # ── Machine-readable dump ──
    RIGOR_JSON.write_text(json.dumps({
        "config":       {"seed": SEED, "threshold": threshold, "n_bootstrap": N_BOOTSTRAP},
        "test_metrics": test_metrics,
        "cv_results":   cv_df.to_dict(orient="records"),
        "delong":       {"aucs": delong_aucs, "pairwise": delong_pw},
        "calibration":  {"raw": cal_raw, "platt_applied": platt, "after_platt": cal_after},
        "confusion_matrices": {
            "default_0.5":                  {"fn": int(fn_d), "fp": int(fp_d)},
            f"tuned_{threshold}":           {"fn": int(fn_t), "fp": int(fp_t)},
        },
    }, indent=2, default=str))

    # ── Console summary ──
    print("\n" + "=" * 72)
    print("HormoBench-PCOS — Scientific Rigor Summary")
    print("=" * 72)
    print(f"Winner: {best_name} | τ={threshold} | bootstrap n={N_BOOTSTRAP}")
    print("\nTest set metrics (point [95% CI]):")
    for k in ["roc_auc","pr_auc","brier","recall","specificity","precision","f1","balanced_accuracy","mcc"]:
        m = test_metrics[k]
        print(f"  {k:20s}  {m['point']:.3f}  [{m['lo']:.3f}, {m['hi']:.3f}]")
    print(f"\n✓ Consolidated write-up → {DOCS}/evaluation_section.md")
    print(f"✓ Machine-readable dump → {RIGOR_JSON}")


if __name__ == "__main__":
    main()
