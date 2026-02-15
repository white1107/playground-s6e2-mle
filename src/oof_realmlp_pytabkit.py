"""RealMLP OOF generator using pytabkit (RealMLP_TD_Classifier).

Based on high-scoring Kaggle kernel approach.
Uses original dataset statistics as features + pytabkit's RealMLP.

Usage:
    python -m src.oof_realmlp_pytabkit
    python -m src.oof_realmlp_pytabkit --n-ens 8 --epochs 100
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.feature_engineering import add_original_statistics, load_original_data


def main():
    parser = argparse.ArgumentParser(description="RealMLP OOF via pytabkit")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ens", type=int, default=8, help="Ensemble size per fold")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load data ---
    train = pd.read_csv("data/train.csv")
    original = load_original_data()

    le = LabelEncoder()
    train["Heart Disease"] = le.fit_transform(train["Heart Disease"])
    original["Heart Disease"] = le.fit_transform(original["Heart Disease"])

    base_features = [c for c in train.columns if c not in ["Heart Disease", "id"]]

    # Add original statistics features
    train = add_original_statistics(train, original, base_features)

    X = train.drop(["id", "Heart Disease"], axis=1)
    y = train["Heart Disease"]

    # Convert all to categorical (as in the reference notebook)
    for col in X.columns:
        X[col] = X[col].astype(str).astype("category")

    print(f"Features: {X.shape[1]}, Samples: {len(X)}")

    # --- Params from high-scoring kernel ---
    param_grid = {
        "device": device,
        "random_state": args.seed,
        "verbosity": 0,
        "n_epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_ens": args.n_ens,
        "use_early_stopping": True,
        "early_stopping_additive_patience": 20,
        "early_stopping_multiplicative_patience": 1,
        "act": "mish",
        "embedding_size": 8,
        "first_layer_lr_factor": 0.5962121993798933,
        "hidden_sizes": "rectangular",
        "hidden_width": 384,
        "lr": 0.04,
        "ls_eps": 0.011498317194338772,
        "ls_eps_sched": "coslog4",
        "max_one_hot_cat_size": 18,
        "n_hidden_layers": 4,
        "p_drop": 0.07301419697186451,
        "p_drop_sched": "flat_cos",
        "plr_hidden_1": 16,
        "plr_hidden_2": 8,
        "plr_lr_factor": 0.1151437622270563,
        "plr_sigma": 2.3316811282666916,
        "scale_lr_factor": 2.244801835541429,
        "sq_mom": 1.0 - 0.011834054955582318,
        "wd": 0.02369230879235962,
    }

    # --- K-Fold ---
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof_preds = np.zeros(len(train))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{args.folds} ---")

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RealMLP_TD_Classifier(**param_grid)
        model.fit(X_tr, y_tr.values, X_val, y_val.values)

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs

        score = roc_auc_score(y_val, val_probs)
        fold_scores.append(score)
        print(f"  Fold {fold + 1} AUC: {score:.5f}")

        if device == "cuda":
            torch.cuda.empty_cache()

    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    print(f"\n{'='*60}")
    print(f"RealMLP CV AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f"Overall OOF AUC: {roc_auc_score(y, oof_preds):.5f}")
    print(f"{'='*60}")

    # Save OOF
    os.makedirs("output/predictions", exist_ok=True)
    oof_path = "output/predictions/oof_realmlp_eng.csv"
    pd.DataFrame({"id": train["id"], "oof_pred": oof_preds}).to_csv(oof_path, index=False)
    print(f"OOF saved: {oof_path}")


if __name__ == "__main__":
    main()
