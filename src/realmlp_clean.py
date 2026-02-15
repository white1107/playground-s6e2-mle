"""Clean RealMLP - exact replica of public notebook (LB: 0.95397).

Usage:
    python -m src.realmlp_clean
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pytabkit import RealMLP_TD_Classifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_STATE = 42
N_FOLDS = 5


def main():
    print(f"Using device: {DEVICE}")

    # --- Data Loading (exact replica of notebook) ---
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    original = pd.read_csv("data/external/Heart_Disease_Prediction.csv")

    le = LabelEncoder()
    train['Heart Disease'] = le.fit_transform(train['Heart Disease'])
    original['Heart Disease'] = le.fit_transform(original['Heart Disease'])

    base_features = [col for col in train.columns if col not in ['Heart Disease', 'id']]

    def add_engineered_features(df):
        df_temp = df.copy()
        for col in base_features:
            if col in original.columns:
                stats = original.groupby(col)['Heart Disease'].agg(
                    ['mean', 'median', 'std', 'skew', 'count']
                ).reset_index()
                stats.columns = [col] + [f"orig_{col}_{s}" for s in ['mean', 'median', 'std', 'skew', 'count']]
                df_temp = df_temp.merge(stats, on=col, how='left')
                fill_values = {
                    f"orig_{col}_mean": original['Heart Disease'].mean(),
                    f"orig_{col}_median": original['Heart Disease'].median(),
                    f"orig_{col}_std": 0,
                    f"orig_{col}_skew": 0,
                    f"orig_{col}_count": 0
                }
                df_temp = df_temp.fillna(value=fill_values)
        return df_temp

    train = add_engineered_features(train)
    test = add_engineered_features(test)

    X = train.drop(['id', 'Heart Disease'], axis=1)
    y = train['Heart Disease']
    X_test = test.drop(['id'], axis=1)

    print(f"X shape: {X.shape}, X_test shape: {X_test.shape}")
    print(f"X columns: {list(X.columns[:5])} ... ({len(X.columns)} total)")

    # --- Convert all to categorical ---
    for col in X.columns:
        X[col] = X[col].astype(str).astype('category')
        X_test[col] = X_test[col].astype(str).astype('category')

    # --- Model params (exact copy from notebook) ---
    param_grid = {
        'device': DEVICE,
        'random_state': 42,
        'verbosity': 2,
        'n_epochs': 100,
        'batch_size': 256,
        'n_ens': 8,
        'use_early_stopping': True,
        'early_stopping_additive_patience': 20,
        'early_stopping_multiplicative_patience': 1,
        'act': "mish",
        'embedding_size': 8,
        'first_layer_lr_factor': 0.5962121993798933,
        'hidden_sizes': "rectangular",
        'hidden_width': 384,
        'lr': 0.04,
        'ls_eps': 0.011498317194338772,
        'ls_eps_sched': "coslog4",
        'max_one_hot_cat_size': 18,
        'n_hidden_layers': 4,
        'p_drop': 0.07301419697186451,
        'p_drop_sched': "flat_cos",
        'plr_hidden_1': 16,
        'plr_hidden_2': 8,
        'plr_lr_factor': 0.1151437622270563,
        'plr_sigma': 2.3316811282666916,
        'scale_lr_factor': 2.244801835541429,
        'sq_mom': 1.0 - 0.011834054955582318,
        'wd': 0.02369230879235962,
    }

    # --- K-Fold CV + Test predictions ---
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Starting Fold {fold + 1} ---")

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RealMLP_TD_Classifier(**param_grid)
        model.fit(X_tr, y_tr.values, X_val, y_val.values)

        val_probs = model.predict_proba(X_val)[:, 1]
        fold_test_probs = model.predict_proba(X_test)[:, 1]

        oof_preds[val_idx] = val_probs
        test_preds += fold_test_probs / N_FOLDS

        score = roc_auc_score(y_val, val_probs)
        fold_scores.append(score)
        print(f"Fold {fold + 1} ROC-AUC Score: {score:.5f}")

        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    total_oof_score = roc_auc_score(y, oof_preds)

    print("\n" + "=" * 40)
    print(f"Overall OOF ROC-AUC: {total_oof_score:.5f}")
    print(f"Mean Fold Score: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print("=" * 40)

    # --- Save ---
    os.makedirs("output/predictions", exist_ok=True)
    os.makedirs("output/submissions", exist_ok=True)

    pd.DataFrame({'id': train['id'], 'Heart Disease_prob': oof_preds}).to_csv(
        'output/predictions/oof_realmlp_clean.csv', index=False
    )

    submission = pd.DataFrame({'id': test['id'], 'Heart Disease': test_preds})
    submission.to_csv('output/submissions/submission_realmlp_clean.csv', index=False)
    print(f"Saved: output/submissions/submission_realmlp_clean.csv ({len(submission)} rows)")
    print(f"Test pred stats: mean={test_preds.mean():.6f}, std={test_preds.std():.6f}")
    print(f"Test pred range: [{test_preds.min():.6f}, {test_preds.max():.6f}]")


if __name__ == "__main__":
    main()
