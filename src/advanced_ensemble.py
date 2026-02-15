"""Advanced ensemble techniques: rank averaging, pseudo-labeling, diversity blending.

Usage:
    # Rank-averaged ensemble (optimize on OOF)
    python -m src.advanced_ensemble rank

    # Pseudo-labeling: generate augmented training data
    python -m src.advanced_ensemble pseudo --threshold 0.95

    # Full pipeline: rank avg + pseudo-label retrain
    python -m src.advanced_ensemble full
"""

import argparse
import os

import numpy as np
import optuna
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# ============================================================================
# Config
# ============================================================================

OOF_DIR = "output/predictions"
SUB_DIR = "output/submissions"

MODELS = {
    "cat": "oof_cat_eng.csv",
    "cat_sa": "oof_cat_seedavg_eng.csv",
    "xgb": "oof_xgb_eng.csv",
    "lgbm": "oof_lgbm_eng.csv",
    "realmlp": "oof_realmlp_eng.csv",
    "tabnet": "oof_tabnet_eng.csv",
    "ft_transformer": "oof_ft_transformer_eng.csv",
}

SUB_PATTERNS = {
    "cat": "submission_cat_eng_kfold.csv",
    "cat_sa": "submission_cat_seedavg_eng_kfold.csv",
    "xgb": "submission_xgb_eng_kfold.csv",
    "lgbm": "submission_lgbm_eng_kfold.csv",
    "realmlp": "submission_realmlp_eng_kfold.csv",
    "tabnet": "submission_tabnet_eng_kfold.csv",
    "ft_transformer": "submission_ft_transformer_eng_kfold.csv",
}


def load_target(use_external=False):
    train = pd.read_csv("data/train.csv")
    if use_external:
        ext_path = "data/external/Heart_Disease_Prediction.csv"
        if os.path.exists(ext_path):
            ext = pd.read_csv(ext_path)
            train = pd.concat([train, ext], axis=0).reset_index(drop=True)
    if train["Heart Disease"].dtype == object:
        return train["Heart Disease"].map({"Presence": 1, "Absence": 0}).values
    return train["Heart Disease"].values


def load_oofs(model_names=None):
    if model_names is None:
        model_names = list(MODELS.keys())
    oofs = {}
    for m in model_names:
        path = os.path.join(OOF_DIR, MODELS[m])
        if os.path.exists(path):
            oofs[m] = pd.read_csv(path)["oof_pred"].values
    return oofs


def load_subs(model_names=None):
    if model_names is None:
        model_names = list(SUB_PATTERNS.keys())
    subs = {}
    for m in model_names:
        path = os.path.join(SUB_DIR, SUB_PATTERNS[m])
        if os.path.exists(path):
            subs[m] = pd.read_csv(path)["Heart Disease"].values
    return subs


# ============================================================================
# 1. Rank Averaging
# ============================================================================

def to_ranks(preds: np.ndarray) -> np.ndarray:
    """Convert predictions to percentile ranks (0-1)."""
    return rankdata(preds) / len(preds)


def rank_blend(preds_dict: dict, weights: dict) -> np.ndarray:
    """Blend predictions using rank averaging with weights."""
    names = sorted(preds_dict.keys())
    total_w = sum(weights.get(n, 0) for n in names)
    blended = np.zeros(len(next(iter(preds_dict.values()))))
    for n in names:
        w = weights.get(n, 0) / total_w
        blended += w * to_ranks(preds_dict[n])
    return blended


def optimize_rank_blend(oofs: dict, y_true: np.ndarray, n_trials: int = 500):
    """Optimize rank-blend weights using Optuna."""
    names = sorted(oofs.keys())
    ranks = {n: to_ranks(oofs[n]) for n in names}

    def objective(trial):
        weights = {}
        for n in names:
            weights[n] = trial.suggest_float(n, 0.0, 1.0)
        total = sum(weights.values())
        if total < 1e-8:
            return 0.0
        blended = sum(weights[n] / total * ranks[n] for n in names)
        return roc_auc_score(y_true, blended)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_w = study.best_params
    total = sum(best_w.values())
    best_w = {k: v / total for k, v in best_w.items()}

    print(f"\nRank Blend Best AUC: {study.best_value:.5f}")
    print("Weights:")
    for n in names:
        print(f"  {n:>16s}: {best_w[n]:.3f}")

    return best_w, study.best_value


def optimize_prob_blend(oofs: dict, y_true: np.ndarray, n_trials: int = 500):
    """Optimize probability-blend weights using Optuna (for comparison)."""
    names = sorted(oofs.keys())

    def objective(trial):
        weights = {}
        for n in names:
            weights[n] = trial.suggest_float(n, 0.0, 1.0)
        total = sum(weights.values())
        if total < 1e-8:
            return 0.0
        blended = sum(weights[n] / total * oofs[n] for n in names)
        return roc_auc_score(y_true, blended)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_w = study.best_params
    total = sum(best_w.values())
    best_w = {k: v / total for k, v in best_w.items()}

    print(f"\nProb Blend Best AUC: {study.best_value:.5f}")
    print("Weights:")
    for n in names:
        print(f"  {n:>16s}: {best_w[n]:.3f}")

    return best_w, study.best_value


# ============================================================================
# 2. Pseudo-Labeling
# ============================================================================

def generate_pseudo_labels(
    test_preds: np.ndarray,
    test_df: pd.DataFrame,
    threshold_high: float = 0.95,
    threshold_low: float = 0.05,
) -> pd.DataFrame:
    """Generate pseudo-labels from high-confidence test predictions.

    Returns a DataFrame with the same columns as training data,
    containing only high-confidence samples.
    """
    mask_pos = test_preds >= threshold_high
    mask_neg = test_preds <= threshold_low
    mask = mask_pos | mask_neg

    pseudo = test_df[mask].copy()
    pseudo["Heart Disease"] = np.where(
        test_preds[mask] >= threshold_high, "Presence", "Absence"
    )

    n_pos = mask_pos.sum()
    n_neg = mask_neg.sum()
    print(f"Pseudo-labels: {len(pseudo)} samples "
          f"({n_pos} positive, {n_neg} negative) "
          f"from {len(test_df)} test rows")
    print(f"  Thresholds: pos >= {threshold_high}, neg <= {threshold_low}")

    return pseudo


def retrain_with_pseudo(
    model_name: str,
    pseudo_df: pd.DataFrame,
    n_folds: int = 5,
    n_trials: int = 50,
    seed: int = 42,
):
    """Retrain a GBDT model with pseudo-labeled data added to training.

    Returns OOF AUC on original data only (not pseudo).
    """
    from src.feature_engineering import (
        add_domain_features,
        add_original_statistics,
        load_original_data,
    )
    from src.oof_generator import generate_oof, save_oof, save_submission
    from src.train import get_feature_names, load_data, TARGET

    # Load original training data
    df_train, df_test = load_data(
        use_engineered=True,
        use_external=True,
        use_original_stats=True,
        validate_schema=False,
    )

    n_orig = len(df_train)

    # Prepare pseudo data with same features
    pseudo_eng = add_domain_features(pseudo_df.copy())
    original = load_original_data()
    base_feats = [c for c in pseudo_df.columns if c not in ["Heart Disease", "id"]]
    pseudo_eng = add_original_statistics(pseudo_eng, original, base_feats)

    # Combine
    df_combined = pd.concat([df_train, pseudo_eng], axis=0).reset_index(drop=True)
    print(f"Combined training: {n_orig} orig + {len(pseudo_eng)} pseudo = {len(df_combined)}")

    # Prepare
    if df_combined[TARGET].dtype == object:
        y = df_combined[TARGET].map({"Presence": 1, "Absence": 0})
    else:
        y = df_combined[TARGET]

    numerical_features, categorical_features = get_feature_names(True, True, df_combined)
    all_features = numerical_features + categorical_features
    X = df_combined[all_features]
    X_test = df_test[all_features] if df_test is not None else None

    native = model_name == "cat"
    result = generate_oof(
        model_name=model_name, X=X, y=y, X_test=X_test,
        n_folds=n_folds, seed=seed, use_engineered=True,
        use_original_stats=True, native_cats=native,
        tune=True, n_trials=n_trials, df_sample=df_combined.head(),
    )

    # Evaluate on original data only
    oof_orig = result["oof_preds"][:n_orig]
    y_orig = y.values[:n_orig]
    orig_auc = roc_auc_score(y_orig, oof_orig)
    print(f"\nPseudo-label {model_name} OOF AUC (orig only): {orig_auc:.5f}")

    suffix = f"_pseudo_{model_name}"
    save_oof(model_name, df_combined["id"], result["oof_preds"], suffix)
    if result["test_preds"] is not None and df_test is not None:
        save_submission(model_name, df_test["id"], result["test_preds"], suffix)

    return orig_auc, result


# ============================================================================
# 3. Create final submission
# ============================================================================

def create_ensemble_submission(weights: dict, use_rank: bool = True):
    """Create final submission from all available test predictions."""
    subs = load_subs(list(weights.keys()))
    available = {k: v for k, v in weights.items() if k in subs}

    if not available:
        print("No submission files found!")
        return

    missing = set(weights.keys()) - set(subs.keys())
    if missing:
        print(f"Warning: Missing submissions for {missing}")

    # Renormalize weights
    total = sum(available.values())
    available = {k: v / total for k, v in available.items()}

    if use_rank:
        blended = rank_blend(subs, available)
    else:
        names = sorted(available.keys())
        blended = sum(available[n] * subs[n] for n in names)

    test = pd.read_csv("data/test.csv")
    os.makedirs(SUB_DIR, exist_ok=True)
    tag = "rank" if use_rank else "prob"
    path = os.path.join(SUB_DIR, f"submission_ensemble_{tag}.csv")
    pd.DataFrame({"id": test["id"], "Heart Disease": blended}).to_csv(path, index=False)
    print(f"Saved: {path} ({len(test)} rows)")
    return path


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Advanced ensemble")
    parser.add_argument("mode", choices=["rank", "prob", "pseudo", "full", "submit"])
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Pseudo-label confidence threshold")
    parser.add_argument("--pseudo-model", type=str, default="cat",
                        help="Model to retrain with pseudo-labels")
    parser.add_argument("--pseudo-trials", type=int, default=50)
    args = parser.parse_args()

    y_true = load_target()
    oofs = load_oofs()
    print(f"Loaded {len(oofs)} OOF files: {list(oofs.keys())}")

    if args.mode == "rank":
        best_w, best_auc = optimize_rank_blend(oofs, y_true, args.trials)
        # Also run prob blend for comparison
        print("\n--- Probability blend for comparison ---")
        prob_w, prob_auc = optimize_prob_blend(oofs, y_true, args.trials)
        print(f"\nRank AUC: {best_auc:.5f} vs Prob AUC: {prob_auc:.5f}")

    elif args.mode == "prob":
        best_w, best_auc = optimize_prob_blend(oofs, y_true, args.trials)

    elif args.mode == "pseudo":
        # Use current best blend to generate pseudo-labels
        subs = load_subs()
        if not subs:
            print("No test submissions found yet. Run after test predictions complete.")
            return
        # Simple average for pseudo-label source
        names = sorted(subs.keys())
        avg_pred = np.mean([subs[n] for n in names], axis=0)
        test_df = pd.read_csv("data/test.csv")
        pseudo = generate_pseudo_labels(
            avg_pred, test_df,
            threshold_high=args.threshold,
            threshold_low=1 - args.threshold,
        )
        pseudo.to_csv("data/pseudo_labels.csv", index=False)
        print(f"Saved pseudo-labels to data/pseudo_labels.csv")

        # Retrain
        auc, _ = retrain_with_pseudo(
            args.pseudo_model, pseudo,
            n_trials=args.pseudo_trials,
        )

    elif args.mode == "submit":
        # Quick: create submissions with both rank and prob blend
        best_w_rank, _ = optimize_rank_blend(oofs, y_true, args.trials)
        best_w_prob, _ = optimize_prob_blend(oofs, y_true, args.trials)
        create_ensemble_submission(best_w_rank, use_rank=True)
        create_ensemble_submission(best_w_prob, use_rank=False)

    elif args.mode == "full":
        # Step 1: Rank blend optimization
        print("=" * 60)
        print("Step 1: Rank blend optimization")
        print("=" * 60)
        rank_w, rank_auc = optimize_rank_blend(oofs, y_true, args.trials)

        # Step 2: Prob blend optimization
        print("\n" + "=" * 60)
        print("Step 2: Probability blend optimization")
        print("=" * 60)
        prob_w, prob_auc = optimize_prob_blend(oofs, y_true, args.trials)

        # Step 3: Create submissions
        print("\n" + "=" * 60)
        print("Step 3: Create ensemble submissions")
        print("=" * 60)
        create_ensemble_submission(rank_w, use_rank=True)
        create_ensemble_submission(prob_w, use_rank=False)

        print(f"\nFinal: Rank AUC={rank_auc:.5f}, Prob AUC={prob_auc:.5f}")


if __name__ == "__main__":
    main()
