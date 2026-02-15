"""Fold-aware Target Encoding for categorical features.

Computes smoothed target statistics from 630K training rows (per fold)
instead of relying on the noisy 303-row original dataset statistics.

Usage:
    python -m src.target_encoding --model cat --trials 100
    python -m src.target_encoding --model cat,xgb,lgbm --trials 50
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.feature_engineering import add_domain_features, add_original_statistics, load_original_data


# ============================================================================
# Target Encoding
# ============================================================================

def compute_target_encoding(
    train_df: pd.DataFrame,
    target: np.ndarray,
    columns: list[str],
    smoothing: float = 20.0,
) -> dict:
    """Compute smoothed target encoding maps from training data.

    Uses Bayesian smoothing: TE = (count * mean + prior * global_mean) / (count + prior)
    """
    global_mean = target.mean()
    te_maps = {}

    for col in columns:
        agg = pd.DataFrame({"target": target, "col": train_df[col].values})
        stats = agg.groupby("col")["target"].agg(["mean", "count"])
        # Bayesian smoothing
        smooth_mean = (
            stats["count"] * stats["mean"] + smoothing * global_mean
        ) / (stats["count"] + smoothing)
        te_maps[col] = {
            "encoding": smooth_mean.to_dict(),
            "global_mean": global_mean,
        }

    return te_maps


def apply_target_encoding(
    df: pd.DataFrame,
    te_maps: dict,
    prefix: str = "te_",
) -> pd.DataFrame:
    """Apply target encoding maps to a DataFrame."""
    df = df.copy()
    for col, info in te_maps.items():
        encoding = info["encoding"]
        global_mean = info["global_mean"]
        df[f"{prefix}{col}_mean"] = df[col].map(encoding).fillna(global_mean)
    return df


def compute_advanced_te(
    train_df: pd.DataFrame,
    target: np.ndarray,
    columns: list[str],
    smoothing: float = 20.0,
) -> dict:
    """Compute advanced target encoding: mean, std, count, positive rate."""
    global_mean = target.mean()
    global_std = target.std()
    n_total = len(target)
    te_maps = {}

    for col in columns:
        agg = pd.DataFrame({"target": target, "col": train_df[col].values})
        stats = agg.groupby("col")["target"].agg(["mean", "std", "count", "sum"])
        stats["std"] = stats["std"].fillna(0)

        # Smoothed mean
        smooth_mean = (
            stats["count"] * stats["mean"] + smoothing * global_mean
        ) / (stats["count"] + smoothing)

        # Smoothed std
        smooth_std = (
            stats["count"] * stats["std"] + smoothing * global_std
        ) / (stats["count"] + smoothing)

        # Frequency (count / total)
        freq = stats["count"] / n_total

        te_maps[col] = {
            "mean": smooth_mean.to_dict(),
            "std": smooth_std.to_dict(),
            "freq": freq.to_dict(),
            "global_mean": global_mean,
            "global_std": global_std,
        }

    return te_maps


def apply_advanced_te(
    df: pd.DataFrame,
    te_maps: dict,
    prefix: str = "te_",
) -> pd.DataFrame:
    """Apply advanced target encoding maps."""
    df = df.copy()
    for col, info in te_maps.items():
        df[f"{prefix}{col}_mean"] = df[col].map(info["mean"]).fillna(info["global_mean"])
        df[f"{prefix}{col}_std"] = df[col].map(info["std"]).fillna(info["global_std"])
        df[f"{prefix}{col}_freq"] = df[col].map(info["freq"]).fillna(0)
    return df


# ============================================================================
# 2-level interaction target encoding
# ============================================================================

def compute_interaction_te(
    train_df: pd.DataFrame,
    target: np.ndarray,
    col_pairs: list[tuple[str, str]],
    smoothing: float = 50.0,
) -> dict:
    """Target encoding on pairs of categorical features."""
    global_mean = target.mean()
    te_maps = {}

    for col_a, col_b in col_pairs:
        key = f"{col_a}_x_{col_b}"
        combined = train_df[col_a].astype(str) + "_" + train_df[col_b].astype(str)
        agg = pd.DataFrame({"target": target, "col": combined.values})
        stats = agg.groupby("col")["target"].agg(["mean", "count"])
        smooth_mean = (
            stats["count"] * stats["mean"] + smoothing * global_mean
        ) / (stats["count"] + smoothing)
        te_maps[key] = {
            "encoding": smooth_mean.to_dict(),
            "global_mean": global_mean,
            "col_a": col_a,
            "col_b": col_b,
        }

    return te_maps


def apply_interaction_te(
    df: pd.DataFrame,
    te_maps: dict,
    prefix: str = "te_",
) -> pd.DataFrame:
    """Apply interaction target encoding."""
    df = df.copy()
    for key, info in te_maps.items():
        combined = df[info["col_a"]].astype(str) + "_" + df[info["col_b"]].astype(str)
        df[f"{prefix}{key}"] = combined.map(info["encoding"]).fillna(info["global_mean"])
    return df


# ============================================================================
# Full pipeline: OOF with target encoding features
# ============================================================================

CAT_FEATURES = [
    "Sex", "Chest pain type", "FBS over 120", "EKG results",
    "Exercise angina", "Slope of ST", "Thallium",
]

# High-signal interaction pairs (based on domain knowledge)
INTERACTION_PAIRS = [
    ("Chest pain type", "Thallium"),
    ("Exercise angina", "Slope of ST"),
    ("Number of vessels fluro", "Thallium"),
    ("Sex", "Chest pain type"),
    ("Exercise angina", "ST depression"),
    ("Age_Bin", "Chest pain type"),
    ("Age_Bin", "Thallium"),
]

NUM_FEATURES = [
    "Age", "BP", "Cholesterol", "Max HR", "ST depression",
    "Number of vessels fluro",
]


def run_te_oof(
    model_name: str = "cat",
    n_folds: int = 5,
    seed: int = 42,
    tune: bool = True,
    n_trials: int = 50,
    smoothing: float = 20.0,
):
    """Run OOF generation with fold-aware target encoding."""
    from src.train import get_feature_names, get_pipeline, optimize_hyperparameters

    print("=" * 60)
    print(f"Target Encoding OOF: {model_name.upper()}")
    print(f"Folds: {n_folds} | Tune: {tune} | Smoothing: {smoothing}")
    print("=" * 60)

    # Load data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    original = load_original_data()

    if train["Heart Disease"].dtype == object:
        y = train["Heart Disease"].map({"Presence": 1, "Absence": 0}).values
    else:
        y = train["Heart Disease"].values

    # Base features: domain + original stats (keep these too)
    base_features = [c for c in train.columns if c not in ["Heart Disease", "id"]]
    train = add_original_statistics(train, original, base_features)
    test = add_original_statistics(test, original, base_features)
    train = add_domain_features(train)
    test = add_domain_features(test)

    # Columns for target encoding (including Age_Bin from domain features)
    te_cols = CAT_FEATURES + ["Age_Bin", "Number of vessels fluro"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(train))
    test_preds_sum = np.zeros(len(test))
    fold_scores = []
    best_params = None

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Compute TE from training fold only
        te_maps = compute_advanced_te(
            train.iloc[tr_idx], y[tr_idx], te_cols, smoothing
        )
        interaction_maps = compute_interaction_te(
            train.iloc[tr_idx], y[tr_idx], INTERACTION_PAIRS, smoothing * 2
        )

        # Apply TE
        train_fold = apply_advanced_te(train.iloc[tr_idx].copy(), te_maps)
        val_fold = apply_advanced_te(train.iloc[va_idx].copy(), te_maps)
        test_fold = apply_advanced_te(test.copy(), te_maps)

        train_fold = apply_interaction_te(train_fold, interaction_maps)
        val_fold = apply_interaction_te(val_fold, interaction_maps)
        test_fold = apply_interaction_te(test_fold, interaction_maps)

        # Build feature lists
        te_feat_cols = [c for c in train_fold.columns if c.startswith("te_")]
        orig_feat_cols = [c for c in train_fold.columns if c.startswith("orig_")]
        from src.feature_engineering import get_domain_features
        domain_feats = get_domain_features()

        all_num = NUM_FEATURES + domain_feats + orig_feat_cols + te_feat_cols
        all_features = all_num + CAT_FEATURES

        X_tr = train_fold[all_features]
        X_va = val_fold[all_features]
        X_te = test_fold[all_features]

        # Tune on first fold
        if tune and fold == 0:
            best_params = optimize_hyperparameters(
                model_name, X_tr, pd.Series(y[tr_idx]),
                X_va, pd.Series(y[va_idx]),
                n_trials, True, True,
            )

        # Build model
        result = get_pipeline(
            model_name, best_params, True, True,
            train_fold.head(), native_cats=(model_name == "cat"),
        )

        if isinstance(result, tuple) and result[1] is not None:
            preprocessor, model = result
            X_tr_t = preprocessor.fit_transform(X_tr)
            X_va_t = preprocessor.transform(X_va)

            if model_name == "cat":
                cat_indices = list(range(
                    X_tr_t.shape[1] - len(CAT_FEATURES),
                    X_tr_t.shape[1],
                ))
                # Cast cat features to str for CatBoost
                import scipy.sparse
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

        # Save fold model
        model_dir = f"output/models/{model_name}_te"
        os.makedirs(model_dir, exist_ok=True)
        fold_data = {
            "te_maps": te_maps,
            "interaction_maps": interaction_maps,
        }
        if isinstance(result, tuple) and result[1] is not None:
            fold_data["preprocessor"] = preprocessor
            fold_data["model"] = model
        else:
            fold_data["pipeline"] = pipeline
        joblib.dump(fold_data, f"{model_dir}/fold{fold}.pkl")

        oof_preds[va_idx] = fold_preds
        score = roc_auc_score(y[va_idx], fold_preds)
        fold_scores.append(score)
        print(f"  Fold {fold + 1} AUC: {score:.5f}")
        print(f"  TE features: {len(te_feat_cols)} | Total features: {len(all_features)}")

    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n{'='*60}")
    print(f"CV AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f"Overall OOF AUC: {overall_auc:.5f}")
    print(f"{'='*60}")

    # Save OOF
    os.makedirs("output/predictions", exist_ok=True)
    suffix = f"_te_{model_name}"
    oof_path = f"output/predictions/oof_{model_name}_te_eng.csv"
    pd.DataFrame({"id": train["id"], "oof_pred": oof_preds}).to_csv(oof_path, index=False)
    print(f"OOF saved: {oof_path}")

    # Save submission
    test_preds = test_preds_sum / n_folds
    os.makedirs("output/submissions", exist_ok=True)
    sub_path = f"output/submissions/submission_{model_name}_te_eng_kfold.csv"
    pd.DataFrame({"id": test["id"], "Heart Disease": test_preds}).to_csv(sub_path, index=False)
    print(f"Submission saved: {sub_path} ({len(test)} rows)")

    return {"cv_auc": mean_auc, "std_auc": std_auc, "overall_auc": overall_auc}


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Target Encoding OOF Pipeline")
    parser.add_argument("--model", type=str, default="cat",
                        help="Model: cat, xgb, lgbm (comma-separated)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune", action="store_true", default=True)
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--smoothing", type=float, default=20.0)
    args = parser.parse_args()

    models = [m.strip() for m in args.model.split(",")]
    do_tune = args.tune and not args.no_tune

    for model_name in models:
        run_te_oof(
            model_name=model_name,
            n_folds=args.folds,
            seed=args.seed,
            tune=do_tune,
            n_trials=args.trials,
            smoothing=args.smoothing,
        )


if __name__ == "__main__":
    main()
