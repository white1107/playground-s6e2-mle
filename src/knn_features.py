"""KNN-based features: proportion of positive class among K nearest neighbors.

Creates fundamentally different features from tree-based models.
Uses BallTree for efficiency on 630K rows.

Usage:
    python -m src.knn_features --model cat --trials 50
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import (
    add_domain_features,
    add_original_statistics,
    get_domain_features,
    load_original_data,
)


NUM_FEATURES = [
    "Age", "BP", "Cholesterol", "Max HR", "ST depression",
    "Number of vessels fluro",
]
CAT_FEATURES = [
    "Sex", "Chest pain type", "FBS over 120", "EKG results",
    "Exercise angina", "Slope of ST", "Thallium",
]


def compute_knn_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    k_values: list[int] = [5, 15, 50, 200],
    algorithm: str = "ball_tree",
) -> np.ndarray:
    """Compute KNN features: positive rate among K nearest neighbors."""
    nn = NearestNeighbors(n_neighbors=max(k_values), algorithm=algorithm, n_jobs=-1)
    nn.fit(X_train)

    distances, indices = nn.kneighbors(X_query)

    features = []
    for k in k_values:
        neighbor_labels = y_train[indices[:, :k]]
        # Positive rate
        pos_rate = neighbor_labels.mean(axis=1)
        features.append(pos_rate)
        # Mean distance to K-NN
        mean_dist = distances[:, :k].mean(axis=1)
        features.append(mean_dist)
        # Distance-weighted positive rate
        dists_k = distances[:, :k] + 1e-8
        weights = 1.0 / dists_k
        weighted_rate = (neighbor_labels * weights).sum(axis=1) / weights.sum(axis=1)
        features.append(weighted_rate)

    return np.column_stack(features)


def get_knn_feature_names(k_values: list[int] = [5, 15, 50, 200]) -> list[str]:
    names = []
    for k in k_values:
        names.extend([
            f"knn_{k}_pos_rate",
            f"knn_{k}_mean_dist",
            f"knn_{k}_weighted_rate",
        ])
    return names


def run_knn_oof(
    model_name: str = "cat",
    n_folds: int = 5,
    seed: int = 42,
    tune: bool = True,
    n_trials: int = 50,
    k_values: list[int] = [5, 15, 50, 200],
):
    """Run OOF with KNN features added."""
    from src.train import get_feature_names, get_pipeline, optimize_hyperparameters

    print("=" * 60)
    print(f"KNN Feature OOF: {model_name.upper()}")
    print(f"K values: {k_values}")
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

    # All numerical features for KNN distance computation
    domain_feats = get_domain_features()
    orig_feats = [c for c in train.columns if c.startswith("orig_")]
    knn_input_cols = NUM_FEATURES + domain_feats

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(train))
    test_preds_sum = np.zeros(len(test))
    fold_scores = []
    best_params = None

    knn_feat_names = get_knn_feature_names(k_values)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Scale for KNN
        scaler = StandardScaler()
        X_knn_tr = scaler.fit_transform(train.iloc[tr_idx][knn_input_cols].values)
        X_knn_va = scaler.transform(train.iloc[va_idx][knn_input_cols].values)
        X_knn_te = scaler.transform(test[knn_input_cols].values)

        print(f"  Computing KNN features (k={k_values})...")
        knn_tr = compute_knn_features(X_knn_tr, y[tr_idx], X_knn_tr, k_values)
        knn_va = compute_knn_features(X_knn_tr, y[tr_idx], X_knn_va, k_values)
        knn_te = compute_knn_features(X_knn_tr, y[tr_idx], X_knn_te, k_values)
        print(f"  KNN features computed: {knn_tr.shape[1]} features")

        # Build full feature dataframes
        all_num = NUM_FEATURES + domain_feats + orig_feats
        all_features = all_num + CAT_FEATURES

        X_tr = train.iloc[tr_idx][all_features].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx][all_features].copy().reset_index(drop=True)
        X_te = test[all_features].copy().reset_index(drop=True)

        # Add KNN features
        for i, fn in enumerate(knn_feat_names):
            X_tr[fn] = knn_tr[:, i]
            X_va[fn] = knn_va[:, i]
            X_te[fn] = knn_te[:, i]

        # Tune on first fold
        if tune and fold == 0:
            best_params = optimize_hyperparameters(
                model_name, X_tr, pd.Series(y[tr_idx]),
                X_va, pd.Series(y[va_idx]),
                n_trials, True, True,
            )

        result = get_pipeline(
            model_name, best_params, True, True,
            X_tr.head(), native_cats=(model_name == "cat"),
        )

        if isinstance(result, tuple) and result[1] is not None:
            preprocessor, model = result
            X_tr_t = preprocessor.fit_transform(X_tr)
            X_va_t = preprocessor.transform(X_va)

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
            X_te_t = preprocessor.transform(X_te)
            if scipy.sparse.issparse(X_te_t):
                X_te_t = X_te_t.toarray()
            if model_name == "cat":
                for ci in cat_indices:
                    X_te_t[:, ci] = X_te_t[:, ci].astype(str)
            test_preds_sum += model.predict_proba(X_te_t)[:, 1]
        else:
            pipeline = result[0] if isinstance(result, tuple) else result
            pipeline.fit(X_tr, pd.Series(y[tr_idx]))
            fold_preds = pipeline.predict_proba(X_va)[:, 1]
            test_preds_sum += pipeline.predict_proba(X_te)[:, 1]

        oof_preds[va_idx] = fold_preds
        score = roc_auc_score(y[va_idx], fold_preds)
        fold_scores.append(score)
        print(f"  Fold {fold + 1} AUC: {score:.5f}")

    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n{'='*60}")
    print(f"KNN+{model_name.upper()} CV AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f"Overall OOF AUC: {overall_auc:.5f}")
    print(f"{'='*60}")

    # Save
    os.makedirs("output/predictions", exist_ok=True)
    oof_path = f"output/predictions/oof_{model_name}_knn_eng.csv"
    pd.DataFrame({"id": train["id"], "oof_pred": oof_preds}).to_csv(oof_path, index=False)
    print(f"OOF saved: {oof_path}")

    test_preds = test_preds_sum / n_folds
    os.makedirs("output/submissions", exist_ok=True)
    sub_path = f"output/submissions/submission_{model_name}_knn_eng_kfold.csv"
    pd.DataFrame({"id": test["id"], "Heart Disease": test_preds}).to_csv(sub_path, index=False)
    print(f"Submission saved: {sub_path} ({len(test)} rows)")

    return {"cv_auc": mean_auc, "overall_auc": overall_auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cat")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--k-values", type=str, default="5,15,50,200")
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]
    run_knn_oof(
        model_name=args.model,
        n_folds=args.folds,
        seed=args.seed,
        tune=True,
        n_trials=args.trials,
        k_values=k_values,
    )


if __name__ == "__main__":
    main()
