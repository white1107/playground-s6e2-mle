"""Multi-seed OOF averaging: run the same model with N different seeds and average.

Reduces variance without changing bias.

Usage:
    python -m src.multi_seed --model cat --n-seeds 10 --trials 50
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.feature_engineering import (
    add_domain_features,
    add_original_statistics,
    get_domain_features,
    get_original_stat_features,
    load_original_data,
)
from src.train import (
    TARGET,
    get_feature_names,
    get_pipeline,
    optimize_hyperparameters,
)


NUM_FEATURES = [
    "Age", "BP", "Cholesterol", "Max HR", "ST depression",
    "Number of vessels fluro",
]
CAT_FEATURES = [
    "Sex", "Chest pain type", "FBS over 120", "EKG results",
    "Exercise angina", "Slope of ST", "Thallium",
]


def run_multi_seed(
    model_name: str = "cat",
    n_seeds: int = 10,
    n_folds: int = 5,
    tune: bool = True,
    n_trials: int = 50,
    base_seed: int = 42,
):
    print("=" * 60)
    print(f"Multi-Seed Averaging: {model_name.upper()}")
    print(f"Seeds: {n_seeds} | Folds: {n_folds}")
    print("=" * 60)

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    original = load_original_data()

    if train["Heart Disease"].dtype == object:
        y = train["Heart Disease"].map({"Presence": 1, "Absence": 0}).values
    else:
        y = train["Heart Disease"].values

    base_features = [c for c in train.columns if c not in ["Heart Disease", "id"]]
    train = add_original_statistics(train, original, base_features)
    test = add_original_statistics(test, original, base_features)
    train = add_domain_features(train)
    test = add_domain_features(test)

    domain_feats = get_domain_features()
    orig_feats = get_original_stat_features(train)
    all_num = NUM_FEATURES + domain_feats + orig_feats
    all_features = all_num + CAT_FEATURES

    X = train[all_features]
    X_test = test[all_features]

    # Tune once with the base seed
    best_params = None
    if tune:
        print(f"\nTuning with seed={base_seed}...")
        skf0 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
        for tr_idx, va_idx in skf0.split(X, y):
            best_params = optimize_hyperparameters(
                model_name, X.iloc[tr_idx], pd.Series(y[tr_idx]),
                X.iloc[va_idx], pd.Series(y[va_idx]),
                n_trials, True, True,
            )
            break  # Only first fold

    all_oof = np.zeros((n_seeds, len(train)))
    all_test = np.zeros((n_seeds, len(test)))
    seed_scores = []

    for s_idx in range(n_seeds):
        seed = base_seed + s_idx * 7  # Different seeds
        print(f"\n--- Seed {s_idx + 1}/{n_seeds} (seed={seed}) ---")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(train))
        test_preds_sum = np.zeros(len(test))
        fold_scores = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            result = get_pipeline(
                model_name, best_params, True, True,
                train.head(), native_cats=(model_name == "cat"),
            )

            if isinstance(result, tuple) and result[1] is not None:
                preprocessor, model = result
                X_tr_t = preprocessor.fit_transform(X.iloc[tr_idx])
                X_va_t = preprocessor.transform(X.iloc[va_idx])

                if model_name == "cat":
                    import scipy.sparse
                    cat_indices = list(range(
                        X_tr_t.shape[1] - len(CAT_FEATURES),
                        X_tr_t.shape[1],
                    ))
                    if scipy.sparse.issparse(X_tr_t):
                        X_tr_t = X_tr_t.toarray()
                        X_va_t = X_va_t.toarray()
                    for ci in cat_indices:
                        X_tr_t[:, ci] = X_tr_t[:, ci].astype(str)
                        X_va_t[:, ci] = X_va_t[:, ci].astype(str)
                    model.fit(X_tr_t, y[tr_idx], cat_features=cat_indices)
                else:
                    model.fit(X_tr_t, y[tr_idx])

                fold_preds = model.predict_proba(X_va_t)[:, 1]
                X_te_t = preprocessor.transform(X_test)
                if scipy.sparse.issparse(X_te_t):
                    X_te_t = X_te_t.toarray()
                if model_name == "cat":
                    for ci in cat_indices:
                        X_te_t[:, ci] = X_te_t[:, ci].astype(str)
                test_preds_sum += model.predict_proba(X_te_t)[:, 1]
            else:
                pipeline = result[0] if isinstance(result, tuple) else result
                pipeline.fit(X.iloc[tr_idx], pd.Series(y[tr_idx]))
                fold_preds = pipeline.predict_proba(X.iloc[va_idx])[:, 1]
                test_preds_sum += pipeline.predict_proba(X_test)[:, 1]

            oof_preds[va_idx] = fold_preds
            score = roc_auc_score(y[va_idx], fold_preds)
            fold_scores.append(score)

        seed_auc = roc_auc_score(y, oof_preds)
        seed_scores.append(seed_auc)
        all_oof[s_idx] = oof_preds
        all_test[s_idx] = test_preds_sum / n_folds
        print(f"  Seed {seed} OOF AUC: {seed_auc:.5f} (fold mean: {np.mean(fold_scores):.5f})")

    # Average across seeds
    avg_oof = all_oof.mean(axis=0)
    avg_test = all_test.mean(axis=0)
    avg_auc = roc_auc_score(y, avg_oof)

    print(f"\n{'='*60}")
    print(f"Individual seed AUCs: {[f'{s:.5f}' for s in seed_scores]}")
    print(f"Averaged OOF AUC ({n_seeds} seeds): {avg_auc:.5f}")
    print(f"{'='*60}")

    # Save
    os.makedirs("output/predictions", exist_ok=True)
    oof_path = f"output/predictions/oof_{model_name}_multiseed_eng.csv"
    pd.DataFrame({"id": train["id"], "oof_pred": avg_oof}).to_csv(oof_path, index=False)
    print(f"OOF saved: {oof_path}")

    os.makedirs("output/submissions", exist_ok=True)
    sub_path = f"output/submissions/submission_{model_name}_multiseed_eng_kfold.csv"
    pd.DataFrame({"id": test["id"], "Heart Disease": avg_test}).to_csv(sub_path, index=False)
    print(f"Submission saved: {sub_path} ({len(test)} rows)")

    return avg_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cat")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    run_multi_seed(
        model_name=args.model,
        n_seeds=args.n_seeds,
        n_folds=args.folds,
        tune=True,
        n_trials=args.trials,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
