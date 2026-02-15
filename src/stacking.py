import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# ============================================================================
# Helpers
# ============================================================================

def _load_target(use_external: bool = True) -> np.ndarray:
    """Load the training target, optionally including external data."""
    train_df = pd.read_csv("data/train.csv")
    if use_external:
        ext_path = "data/external/Heart_Disease_Prediction.csv"
        if os.path.exists(ext_path):
            ext_df = pd.read_csv(ext_path)
            train_df = pd.concat([train_df, ext_df], axis=0).reset_index(drop=True)

    if set(train_df["Heart Disease"].unique()) == {"Presence", "Absence"}:
        return train_df["Heart Disease"].map({"Presence": 1, "Absence": 0}).values
    return train_df["Heart Disease"].values


def _load_oof(models: list[str], suffix: str = "_eng") -> dict[str, np.ndarray]:
    """Load OOF prediction files for given models."""
    oof: dict[str, np.ndarray] = {}
    for m in models:
        path = f"output/predictions/oof_{m}{suffix}.csv"
        if os.path.exists(path):
            oof[m] = pd.read_csv(path)["oof_pred"].values
            print(f"  [OOF] Loaded {m}: {path}")
        else:
            print(f"  [OOF] Missing {m}: {path}")
    return oof


def _load_submissions(models: list[str], suffix: str = "_eng") -> dict[str, np.ndarray]:
    """Load test submission files for given models."""
    subs: dict[str, np.ndarray] = {}
    for m in models:
        for pattern in [
            f"output/submissions/submission_{m}{suffix}_kfold.csv",
            f"output/submissions/submission_{m}{suffix}.csv",
        ]:
            if os.path.exists(pattern):
                subs[m] = pd.read_csv(pattern)["Heart Disease"].values
                print(f"  [SUB] Loaded {m}: {pattern}")
                break
        else:
            print(f"  [SUB] Missing {m}")
    return subs


# ============================================================================
# Meta-feature creation
# ============================================================================

def build_meta_features(preds: dict[str, np.ndarray]) -> np.ndarray:
    """Build meta-features from base model predictions.

    For each sample the meta-features are:
      - raw prediction from each model
      - mean across models
      - std across models
      - max - min (spread)
      - rank (per-model percentile)
    """
    names = sorted(preds.keys())
    raw = np.column_stack([preds[n] for n in names])  # (N, M)

    # Statistics across models
    mean = raw.mean(axis=1, keepdims=True)
    std = raw.std(axis=1, keepdims=True)
    spread = raw.max(axis=1, keepdims=True) - raw.min(axis=1, keepdims=True)

    # Rank features (percentile within each model)
    from scipy.stats import rankdata
    ranks = np.column_stack([rankdata(preds[n]) / len(preds[n]) for n in names])

    return np.hstack([raw, mean, std, spread, ranks])


# ============================================================================
# Meta-learners
# ============================================================================

def _get_meta_learner(name: str):
    if name == "lr":
        return LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    if name == "ridge":
        return RidgeClassifier(alpha=1.0)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            num_leaves=8,
            min_child_samples=20,
            verbose=-1,
        )
    raise ValueError(f"Unknown meta-learner: {name}")


def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Unified predict_proba that works for both probabilistic and
    decision-function models (e.g. RidgeClassifier)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # RidgeClassifier: use decision_function -> sigmoid
    d = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))


# ============================================================================
# Main stacking routine
# ============================================================================

def run_stacking(
    models: list[str],
    meta_learner_name: str = "lr",
    n_folds: int = 5,
    suffix: str = "_eng",
    use_external: bool = True,
    seed: int = 42,
):
    print("=" * 60)
    print("Stacking Ensemble")
    print(f"Base models : {models}")
    print(f"Meta-learner: {meta_learner_name}")
    print(f"Folds       : {n_folds}")
    print("=" * 60)

    # --- Load target ---
    y_true = _load_target(use_external)

    # --- Load OOF predictions ---
    print("\nLoading OOF predictions...")
    oof_preds = _load_oof(models, suffix)
    active = [m for m in models if m in oof_preds]
    if len(active) < 2:
        print("Error: Need at least 2 OOF files for stacking.")
        return
    oof_preds = {m: oof_preds[m] for m in active}

    # Verify lengths match
    for m, v in oof_preds.items():
        if len(v) != len(y_true):
            print(f"Error: OOF length mismatch for {m}: {len(v)} vs {len(y_true)}")
            return

    # --- Build meta-features ---
    X_meta = build_meta_features(oof_preds)
    print(f"\nMeta-features shape: {X_meta.shape}  ({len(active)} models)")

    # --- K-Fold meta-training ---
    print(f"\nTraining meta-model ({meta_learner_name}) with {n_folds}-fold CV...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    meta_oof = np.zeros(len(y_true))
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_meta, y_true)):
        X_tr, X_va = X_meta[tr_idx], X_meta[va_idx]
        y_tr, y_va = y_true[tr_idx], y_true[va_idx]

        meta_model = _get_meta_learner(meta_learner_name)
        meta_model.fit(X_tr, y_tr)

        preds = _predict_proba(meta_model, X_va)
        meta_oof[va_idx] = preds
        score = roc_auc_score(y_va, preds)
        fold_scores.append(score)
        print(f"  Fold {fold + 1}: AUC = {score:.4f}")

    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    print(f"\nStacked CV AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

    # --- Train final meta-model on all data ---
    final_model = _get_meta_learner(meta_learner_name)
    final_model.fit(X_meta, y_true)

    # Save meta-model
    os.makedirs("output/models", exist_ok=True)
    meta_model_path = "output/models/meta_model_stacked.pkl"
    joblib.dump({"model": final_model, "active_models": active}, meta_model_path)
    print(f"Meta-model saved to {meta_model_path}")

    # Save stacking OOF
    os.makedirs("output/predictions", exist_ok=True)
    oof_df = pd.DataFrame({"oof_pred": meta_oof})
    oof_df.to_csv("output/predictions/oof_stacked.csv", index=False)

    # --- Generate submission ---
    print("\nLoading test submissions...")
    sub_preds = _load_submissions(active, suffix)
    if len(sub_preds) < len(active):
        missing = set(active) - set(sub_preds.keys())
        print(f"Warning: Missing test submissions for {missing}. Skipping submission.")
        return

    X_sub = build_meta_features(sub_preds)
    final_preds = _predict_proba(final_model, X_sub)

    test_df = pd.read_csv("data/test.csv")
    submission = pd.DataFrame({
        "id": test_df["id"],
        "Heart Disease": final_preds,
    })

    os.makedirs("output/submissions", exist_ok=True)
    out_path = f"output/submissions/submission_stacked_{meta_learner_name}.csv"
    submission.to_csv(out_path, index=False)
    print(f"Stacked submission saved to {out_path}")

    return {"cv_auc_mean": mean_auc, "cv_auc_std": std_auc}


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stacking Ensemble")
    parser.add_argument(
        "--models",
        type=str,
        default="cat,xgb,lgbm",
        help="Comma-separated base model names (default: cat,xgb,lgbm)",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="lr",
        choices=["lr", "ridge", "lgbm"],
        help="Meta-learner: lr, ridge, lgbm (default: lr)",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--suffix", type=str, default="_eng", help="File suffix (default: _eng)")
    parser.add_argument("--no-external", action="store_true", help="Skip external data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    run_stacking(
        models=args.models.split(","),
        meta_learner_name=args.meta,
        n_folds=args.folds,
        suffix=args.suffix,
        use_external=not args.no_external,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
