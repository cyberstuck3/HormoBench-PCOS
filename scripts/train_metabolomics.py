"""
Metabolomics track, Stage 2: train and evaluate on the metabolomics cohort.

n = 72, so the design differs from the Kerala track:
  * 80/20 stratified split (no separate validation set; a 15 % val set would
    hold ~11 people). Model selection uses 5-fold CV AUC on train only.
  * Decision threshold fixed at 0.50 (too few samples to tune one honestly).
  * Three feature sets, so the contribution of amino acids is visible:
      clinical    : Age, Height, Weight, BMI, testosterone
      amino_acids : 17 follicular-fluid amino acids
      combined    : both
The test set is touched once per feature set, after selection.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from hormobench import META_CLEAN_CSV, META_SPLITS, META_RESULTS, SEED, N_BOOTSTRAP
from hormobench.registry import MODEL_REGISTRY, build_model
from hormobench.rigor.metrics import compute_all_metrics

TARGET = "PCOS (Y/N)"
THRESHOLD = 0.50
CLINICAL = ["Age", "Height (m)", "Weight (kg)", "BMI", "Serum testosterone (ng/ml)"]


def feature_sets(columns):
    amino = [c for c in columns if c.startswith("L-")]
    return {"clinical": CLINICAL, "amino_acids": amino, "combined": CLINICAL + amino}


def main():
    df = pd.read_csv(META_CLEAN_CSV)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    for name, ys in [("Train", y_train), ("Test", y_test)]:
        print(f"  {name:5s}  n={len(ys):3d}   PCOS+ = {ys.sum():3d} ({ys.mean()*100:.1f}%)")

    META_SPLITS.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"seed": SEED, "train_idx": X_train.index.tolist(),
               "test_idx": X_test.index.tolist()}, open(META_SPLITS, "w"))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out = {"n_total": len(df), "n_train": len(y_train), "n_test": len(y_test),
           "threshold": THRESHOLD, "feature_sets": {}}

    for fs_name, cols in feature_sets(X.columns).items():
        print(f"\n=== Feature set: {fs_name} ({len(cols)} features) ===")
        cv_scores, fitted = {}, {}
        for model_name, entry in MODEL_REGISTRY.items():
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", build_model(model_name)),
            ])
            grid = {f"model__{k}": v for k, v in entry["param_grid"].items()}
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            gs.fit(X_train[cols], y_train)
            cv_scores[model_name] = {
                "cv_auc_mean": float(gs.best_score_),
                "cv_auc_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
                "best_params": {k.removeprefix("model__"): v for k, v in gs.best_params_.items()},
            }
            fitted[model_name] = gs.best_estimator_
            print(f"  {model_name:20s} CV AUC {gs.best_score_:.3f} "
                  f"± {cv_scores[model_name]['cv_auc_std']:.3f}")

        best = max(cv_scores, key=lambda k: cv_scores[k]["cv_auc_mean"])
        y_prob = fitted[best].predict_proba(X_test[cols])[:, 1]
        test = compute_all_metrics(y_test.values, y_prob, threshold=THRESHOLD,
                                   n_iter=N_BOOTSTRAP, seed=SEED)
        out["feature_sets"][fs_name] = {
            "features": cols, "best_model": best,
            "cv": cv_scores, "test": test,
        }
        print(f"  → winner {best}: test ROC-AUC {test['roc_auc']['point']:.3f} "
              f"[{test['roc_auc']['lo']:.3f}, {test['roc_auc']['hi']:.3f}]")

    META_RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(META_RESULTS / "results.json", "w"), indent=2, default=float)
    _write_table(out, META_RESULTS / "test_metrics.md")
    print(f"\n✓ Results → {META_RESULTS}")


def _write_table(out, path):
    metrics = ["roc_auc", "pr_auc", "recall", "specificity", "f1", "brier"]
    lines = [
        f"# Metabolomics cohort: test-set metrics (n_test = {out['n_test']}, "
        f"threshold {out['threshold']:.2f}, 95 % bootstrap CI)",
        "",
        "| Feature set | Model | CV AUC | " + " | ".join(metrics) + " |",
        "|---|---|:-:|" + ":-:|" * len(metrics),
    ]
    for fs, r in out["feature_sets"].items():
        cv = r["cv"][r["best_model"]]
        cells = [f"{r['test'][m]['point']:.3f} [{r['test'][m]['lo']:.2f}, "
                 f"{r['test'][m]['hi']:.2f}]" for m in metrics]
        lines.append(f"| {fs} | {r['best_model']} | {cv['cv_auc_mean']:.3f} ± "
                     f"{cv['cv_auc_std']:.3f} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
