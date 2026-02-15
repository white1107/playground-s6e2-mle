"""OOF (Out-Of-Fold) prediction generator.

Dedicated module for generating OOF predictions and fold-averaged test
predictions that stacking / ensemble scripts consume.

Usage:
    # Generate OOF for a single model
    python -m src.oof_generator --model cat --engineered --folds 5

    # Generate OOF for all GBDT models
    python -m src.oof_generator --model cat,xgb,lgbm --engineered --folds 5 --tune --n-trials 50

    # Generate OOF with all bells and whistles
    python -m src.oof_generator --model cat,xgb,lgbm --engineered --original-stats --external --folds 10 --tune
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.feature_engineering import add_domain_features, add_original_statistics
from src.train import (
    get_feature_names,
    get_pipeline,
    load_data,
    optimize_hyperparameters,
    TARGET,
)


# ============================================================================
# Core OOF generation
# ============================================================================

def generate_oof(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame | None = None,
    params: dict | None = None,
    n_folds: int = 5,
    seed: int = 42,
    use_engineered: bool = False,
    use_original_stats: bool = False,
    native_cats: bool = False,
    tune: bool = False,
    n_trials: int = 50,
    df_sample: pd.DataFrame | None = None,
) -> dict:
    """Generate OOF predictions with K-Fold CV.

    Returns:
        dict with keys:
            oof_preds   : np.ndarray of OOF predictions (len == len(X))
            test_preds  : np.ndarray of averaged test predictions (or None)
            fold_scores : list of per-fold AUC scores
            best_params : dict of best hyperparameters (if tuned)
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_preds = np.zeros(len(X))
    test_preds_sum = np.zeros(len(X_test)) if X_test is not None else None
    fold_scores: list[float] = []
    best_params = params

    print(f"\n{'='*60}")
    print(f"OOF Generation: {model_name.upper()}")
    print(f"Folds: {n_folds} | Tune: {tune} | Features: {X.shape[1]}")
    print(f"{'='*60}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # Tune on first fold only
        fold_params = best_params
        if tune and fold == 0:
            fold_params = optimize_hyperparameters(
                model_name, X_tr, y_tr, X_va, y_va,
                n_trials, use_engineered, use_original_stats,
            )
            best_params = fold_params
        elif tune:
            fold_params = best_params

        # Build pipeline / model
        result = get_pipeline(
            model_name, fold_params, use_engineered,
            use_original_stats, df_sample, native_cats=native_cats,
        )

        if isinstance(result, tuple) and result[1] is not None:
            # CatBoost path: separate preprocessor + model
            preprocessor, model = result
            X_tr_t = preprocessor.fit_transform(X_tr)
            X_va_t = preprocessor.transform(X_va)

            if native_cats and model_name == "cat":
                _, cat_feats = get_feature_names(use_engineered, use_original_stats, df_sample)
                cat_indices = list(range(X_tr_t.shape[1] - len(cat_feats), X_tr_t.shape[1]))
                model.fit(X_tr_t, y_tr, cat_features=cat_indices)
            else:
                model.fit(X_tr_t, y_tr)

            fold_preds = model.predict_proba(X_va_t)[:, 1]
            if X_test is not None:
                test_preds_sum += model.predict_proba(preprocessor.transform(X_test))[:, 1]
        else:
            # Pipeline path (RF, LGBM, XGB)
            pipeline = result[0] if isinstance(result, tuple) else result
            pipeline.fit(X_tr, y_tr)

            fold_preds = pipeline.predict_proba(X_va)[:, 1]
            if X_test is not None:
                test_preds_sum += pipeline.predict_proba(X_test)[:, 1]

        # Save fold model for later re-use (predict on new test sets)
        model_dir = f"output/models/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        fold_path = f"{model_dir}/fold{fold}.pkl"
        if isinstance(result, tuple) and result[1] is not None:
            joblib.dump({"preprocessor": preprocessor, "model": model}, fold_path)
        else:
            joblib.dump({"pipeline": pipeline}, fold_path)
        print(f"  Model saved: {fold_path}")

        oof_preds[va_idx] = fold_preds
        score = roc_auc_score(y_va, fold_preds)
        fold_scores.append(score)
        print(f"  Fold {fold + 1} AUC: {score:.4f}")

    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} CV AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"{'='*60}")

    test_preds = test_preds_sum / n_folds if test_preds_sum is not None else None

    return {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "fold_scores": fold_scores,
        "best_params": best_params,
    }


# ============================================================================
# Save helpers
# ============================================================================

def save_oof(
    model_name: str,
    ids: pd.Series,
    oof_preds: np.ndarray,
    suffix: str = "_eng",
) -> str:
    """Save OOF predictions to CSV. Returns the file path."""
    os.makedirs("output/predictions", exist_ok=True)
    path = f"output/predictions/oof_{model_name}{suffix}.csv"
    pd.DataFrame({"id": ids, "oof_pred": oof_preds}).to_csv(path, index=False)
    print(f"OOF saved: {path}")
    return path


def save_submission(
    model_name: str,
    ids: pd.Series,
    test_preds: np.ndarray,
    suffix: str = "_eng",
) -> str:
    """Save test submission to CSV. Returns the file path."""
    os.makedirs("output/submissions", exist_ok=True)
    path = f"output/submissions/submission_{model_name}{suffix}_kfold.csv"
    pd.DataFrame({"id": ids, "Heart Disease": test_preds}).to_csv(path, index=False)
    print(f"Submission saved: {path}")
    return path


# ============================================================================
# Predict from saved fold models (no retraining needed)
# ============================================================================

def predict_from_saved(
    model_name: str,
    X_test: pd.DataFrame,
    n_folds: int = 5,
) -> np.ndarray:
    """Load saved fold models and generate averaged test predictions."""
    model_dir = f"output/models/{model_name}"
    test_preds = np.zeros(len(X_test))

    for fold in range(n_folds):
        fold_path = f"{model_dir}/fold{fold}.pkl"
        if not os.path.exists(fold_path):
            raise FileNotFoundError(f"Saved model not found: {fold_path}")

        saved = joblib.load(fold_path)
        if "pipeline" in saved:
            test_preds += saved["pipeline"].predict_proba(X_test)[:, 1]
        else:
            X_t = saved["preprocessor"].transform(X_test)
            test_preds += saved["model"].predict_proba(X_t)[:, 1]
        print(f"  Fold {fold + 1} loaded & predicted from {fold_path}")

    return test_preds / n_folds


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate OOF predictions for stacking / ensemble",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Comma-separated model names: rf,lgbm,xgb,cat",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--engineered", action="store_true", help="Use domain features")
    parser.add_argument("--original-stats", action="store_true", help="Use original dataset statistics")
    parser.add_argument("--external", action="store_true", help="Include external dataset")
    parser.add_argument("--native-cats", action="store_true", help="CatBoost native categoricals")
    parser.add_argument("--tune", action="store_true", help="Enable Optuna tuning (1st fold)")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials (default: 50)")
    parser.add_argument("--no-validate", action="store_true", help="Skip schema validation")
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip training; load saved fold models and predict on test set")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.model.split(",")]
    suffix = "_eng" if args.engineered else ""

    # Fast path: predict from saved models (no retraining)
    if args.predict_only:
        df_train, df_test = load_data(
            use_engineered=args.engineered,
            use_external=args.external,
            use_original_stats=args.original_stats,
            validate_schema=not args.no_validate,
        )
        numerical_features, categorical_features = get_feature_names(
            args.engineered, args.original_stats, df_train,
        )
        all_features = numerical_features + categorical_features
        X_test = df_test[all_features]
        for model_name in model_names:
            print(f"\nPredicting {model_name} from saved fold models...")
            preds = predict_from_saved(model_name, X_test, args.folds)
            save_submission(model_name, df_test["id"], preds, suffix)
        return

    # Load data once
    df_train, df_test = load_data(
        use_engineered=args.engineered,
        use_external=args.external,
        use_original_stats=args.original_stats,
        validate_schema=not args.no_validate,
    )

    # Prepare target
    if set(df_train[TARGET].unique()) == {"Presence", "Absence"}:
        y = df_train[TARGET].map({"Presence": 1, "Absence": 0})
    else:
        y = df_train[TARGET]

    # Prepare features
    numerical_features, categorical_features = get_feature_names(
        args.engineered, args.original_stats, df_train,
    )
    all_features = numerical_features + categorical_features
    X = df_train[all_features]
    X_test = df_test[all_features] if df_test is not None else None

    # Generate OOF for each model
    results = {}
    for model_name in model_names:
        native = args.native_cats and model_name == "cat"
        result = generate_oof(
            model_name=model_name,
            X=X,
            y=y,
            X_test=X_test,
            n_folds=args.folds,
            seed=args.seed,
            use_engineered=args.engineered,
            use_original_stats=args.original_stats,
            native_cats=native,
            tune=args.tune,
            n_trials=args.n_trials,
            df_sample=df_train.head(),
        )
        results[model_name] = result

        # Save
        save_oof(model_name, df_train["id"], result["oof_preds"], suffix)
        if result["test_preds"] is not None and df_test is not None:
            save_submission(model_name, df_test["id"], result["test_preds"], suffix)

    # Summary
    print(f"\n{'='*60}")
    print("OOF Generation Summary")
    print(f"{'='*60}")
    for name, r in results.items():
        mean = np.mean(r["fold_scores"])
        std = np.std(r["fold_scores"])
        print(f"  {name:>12s}: AUC = {mean:.4f} +/- {std:.4f}")
    print(f"{'='*60}")
    print(f"\nReady for stacking: python -m src.stacking --models {','.join(model_names)}")


if __name__ == "__main__":
    main()
