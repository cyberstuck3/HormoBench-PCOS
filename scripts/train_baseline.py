"""
Stage 2 — Train and tune four candidate classifiers on the training set.

Anti-leakage: preprocessing pipeline is refit inside every CV fold.
The held-out test set is loaded but only used at the very end for the final
metric report; no hyperparameter is tuned against it.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
)
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.metrics         import roc_auc_score, f1_score

from hormobench import (
    CLEAN_CSV, SPLITS_JSON, BASELINE_JSON,
    MODEL_JOBLIB, PREPROC_JOBLIB, SEED,
)
from hormobench.registry import MODEL_REGISTRY

TARGET = "PCOS (Y/N)"


def main():
    # ---- Load + stratified 70/15/15 split ----
    df = pd.read_csv(CLEAN_CSV)
    X, y = df.drop(columns=[TARGET]), df[TARGET]

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, random_state=SEED, stratify=y_tmp
    )

    print("=" * 60)
    print("STRATIFIED SPLIT (70 / 15 / 15)")
    print("=" * 60)
    for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {name:5s}  n={len(ys):3d}   PCOS+ = {ys.sum():3d} ({ys.mean()*100:.1f}%)")

    # ---- Preprocessing fit on train only ----
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    X_train_pp = preprocessor.fit_transform(X_train)
    X_val_pp   = preprocessor.transform(X_val)

    # ---- Persist split indices ----
    splits = {
        "seed":      SEED,
        "train_idx": X_train.index.tolist(),
        "val_idx":   X_val.index.tolist(),
        "test_idx":  X_test.index.tolist(),
    }
    SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(splits, open(SPLITS_JSON, "w"))
    print(f"\n✓ Splits written : {SPLITS_JSON}")

    # ---- Tune each model via 5-fold stratified CV on train ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print("\n" + "=" * 60)
    print("GRID SEARCH  (5-fold stratified CV, scoring=roc_auc)")
    print("=" * 60)

    results, best_models = {}, {}
    for name, entry in MODEL_REGISTRY.items():
        model = entry["cls"](**entry["base_kw"])
        gs = GridSearchCV(model, entry["param_grid"],
                          cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
        gs.fit(X_train_pp, y_train)
        y_val_prob = gs.best_estimator_.predict_proba(X_val_pp)[:, 1]
        y_val_pred = gs.best_estimator_.predict(X_val_pp)
        results[name] = {
            "best_params": gs.best_params_,
            "cv_auc_mean": gs.best_score_,
            "val_auc":     roc_auc_score(y_val, y_val_prob),
            "val_f1":      f1_score(y_val, y_val_pred),
        }
        best_models[name] = gs.best_estimator_
        print(f"\n  {name}")
        print(f"    best_params : {gs.best_params_}")
        print(f"    CV AUC      : {gs.best_score_:.4f}")
        print(f"    Val AUC     : {results[name]['val_auc']:.4f}")
        print(f"    Val F1      : {results[name]['val_f1']:.4f}")

    # ---- Pick winner ----
    best_name = max(results, key=lambda k: results[k]["val_auc"])
    best_model = best_models[best_name]
    print("\n" + "=" * 60)
    print(f"WINNER: {best_name}")
    print("=" * 60)

    # ---- Persist artefacts ----
    MODEL_JOBLIB.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model,   MODEL_JOBLIB)
    joblib.dump(preprocessor, PREPROC_JOBLIB)
    final = {
        "best_model":  best_name,
        "best_params": results[best_name]["best_params"],
        "all_models":  results,
    }
    json.dump(final, open(BASELINE_JSON, "w"), indent=2)
    print(f"\n✓ Model   → {MODEL_JOBLIB}")
    print(f"✓ Preproc → {PREPROC_JOBLIB}")
    print(f"✓ Results → {BASELINE_JSON}")


if __name__ == "__main__":
    main()
