"""TabNet OOF generator with original statistics features.

Usage:
    python -m src.oof_tabnet
    python -m src.oof_tabnet --epochs 200 --patience 30
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.feature_engineering import (
    add_domain_features,
    add_original_statistics,
    get_domain_features,
    load_original_data,
)


def main():
    parser = argparse.ArgumentParser(description="TabNet OOF generator")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()

    print("Loading data for TabNet...")
    df = pd.read_csv("data/train.csv")
    original = load_original_data()

    # Prepare target
    if df["Heart Disease"].dtype == object:
        df["Heart Disease"] = df["Heart Disease"].map({"Presence": 1, "Absence": 0})
    # original already has 0/1 from load_original_data()

    # Features
    num_features = ["Age", "BP", "Cholesterol", "Max HR", "ST depression", "Number of vessels fluro"]
    cat_features = ["Sex", "Chest pain type", "FBS over 120", "EKG results",
                    "Exercise angina", "Slope of ST", "Thallium"]
    base_features = [c for c in df.columns if c not in ["Heart Disease", "id"]]

    # Add original statistics
    print("Adding original statistics...")
    df = add_original_statistics(df, original, base_features)

    # Add domain features
    print("Adding domain features...")
    df = add_domain_features(df)
    domain_feats = get_domain_features()

    # Collect all features
    orig_feat_cols = [c for c in df.columns if c.startswith("orig_")]
    all_num_features = num_features + domain_feats + orig_feat_cols
    all_features = all_num_features + cat_features

    X = df[all_features].copy()
    y = df["Heart Disease"].values

    # Encode categoricals for TabNet
    cat_idxs = []
    cat_dims = []
    for i, col in enumerate(all_features):
        if col in cat_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            cat_idxs.append(i)
            cat_dims.append(len(le.classes_))

    X_values = X.values.astype(np.float32)

    print(f"Features: {len(all_features)} ({len(all_num_features)} num + {len(cat_features)} cat)")
    print(f"Samples: {len(X)}")

    # K-Fold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_preds = np.zeros(len(df))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_values, y)):
        print(f"\n--- Fold {fold + 1}/{args.folds} ---")

        X_train, y_train = X_values[train_idx], y[train_idx]
        X_valid, y_valid = X_values[val_idx], y[val_idx]

        clf = TabNetClassifier(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=4,
            n_d=32,
            n_a=32,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=args.lr),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type="entmax",
            verbose=0,
        )

        clf.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=["valid"],
            eval_metric=["auc"],
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False,
        )

        fold_preds = clf.predict_proba(X_valid)[:, 1]
        oof_preds[val_idx] = fold_preds

        score = roc_auc_score(y_valid, fold_preds)
        fold_scores.append(score)
        print(f"  Fold {fold + 1} AUC: {score:.5f}")

    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n{'='*60}")
    print(f"TabNet CV AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f"Overall OOF AUC: {overall_auc:.5f}")
    print(f"{'='*60}")

    # Save OOF
    os.makedirs("output/predictions", exist_ok=True)
    oof_path = "output/predictions/oof_tabnet_eng.csv"
    pd.DataFrame({"id": df["id"], "oof_pred": oof_preds}).to_csv(oof_path, index=False)
    print(f"OOF saved: {oof_path}")


if __name__ == "__main__":
    main()
