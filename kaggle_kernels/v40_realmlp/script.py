"""V40 RealMLP + Original Dataset Stats - Kaggle Environment."""
import subprocess
subprocess.check_call(['pip', 'install', 'pytabkit', '-q'])

import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from pytabkit import RealMLP_TD_Classifier
import time
import os

warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

class CFG:
    VERSION = "V40"
    PARAM_GRID = {
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
    SEED = 42
    N_FOLDS = 5
    TRAIN_PATH = '/kaggle/input/competitions/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/competitions/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/datasets/rishidamarla/heart-disease-prediction/Heart_Disease_Prediction.csv'

def add_engineered_features(df, original, base_features):
    df_temp = df.copy()
    for col in base_features:
        if col in original.columns:
            stats = original.groupby(col)['Heart Disease'].agg(['mean', 'median', 'std', 'skew', 'count']).reset_index()
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

def main():
    print(f"{'='*80}")
    print(f"S6E2_{CFG.VERSION}_RealMLP (Kaggle Env, device={DEVICE})")
    print(f"{'='*80}")
    start_time = time.time()

    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    original = pd.read_csv(CFG.ORIG_PATH)

    print(f"Train: {train.shape}, Test: {test.shape}, Original: {original.shape}")

    le = LabelEncoder()
    train['Heart Disease'] = le.fit_transform(train['Heart Disease'])
    original['Heart Disease'] = le.fit_transform(original['Heart Disease'])

    base_features = [col for col in train.columns if col not in ['Heart Disease', 'id']]
    train = add_engineered_features(train, original, base_features)
    test = add_engineered_features(test, original, base_features)

    X = train.drop(['id', 'Heart Disease'], axis=1)
    y = train['Heart Disease']
    X_test = test.drop(['id'], axis=1)

    print("Converting all features to categorical type...")
    for col in X.columns:
        X[col] = X[col].astype(str).astype('category')
        X_test[col] = X_test[col].astype(str).astype('category')

    print(f"Total features: {len(X.columns)}")

    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_scores = []

    print(f"\nStarting {CFG.N_FOLDS}-Fold CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold} ---")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RealMLP_TD_Classifier(**CFG.PARAM_GRID)
        model.fit(X_tr, y_tr.values, X_val, y_val.values)

        val_probs = model.predict_proba(X_val)[:, 1]
        fold_test_probs = model.predict_proba(X_test)[:, 1]

        oof_preds[val_idx] = val_probs
        test_preds += fold_test_probs / CFG.N_FOLDS

        score = roc_auc_score(y_val, val_probs)
        fold_scores.append(score)
        print(f"Fold {fold} AUC: {score:.5f}")

        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    overall_score = roc_auc_score(y, oof_preds)
    print(f"\n{'='*40}")
    print(f"Overall OOF AUC: {overall_score:.5f}")
    print(f"Mean Fold: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")

    pd.DataFrame({'id': train['id'], 'Heart Disease_prob': oof_preds}).to_csv("oof_v40.csv", index=False)
    pd.DataFrame({'id': test['id'], 'Heart Disease': test_preds}).to_csv("submission.csv", index=False)

    elapsed = (time.time() - start_time) / 60
    print(f"Saved: submission.csv, oof_v40.csv")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
