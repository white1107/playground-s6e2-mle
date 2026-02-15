"""V48 RealMLP Multi-Seed Ensemble - Kaggle Environment."""
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
from scipy.stats import rankdata
import time
import os

warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

class CFG:
    VERSION = "V48"
    SEEDS = [42, 123, 456, 789, 2026]
    BASE_PARAMS = {
        'device': DEVICE,
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

def train_single_seed(seed, X, y, X_test, n_folds=5):
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Seed {seed} - Fold {fold + 1}/{n_folds}")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        params = CFG.BASE_PARAMS.copy()
        params['random_state'] = seed

        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y_tr.values, X_val, y_val.values)

        val_probs = model.predict_proba(X_val)[:, 1]
        fold_test_probs = model.predict_proba(X_test)[:, 1]

        oof_preds[val_idx] = val_probs
        test_preds += fold_test_probs / n_folds

        score = roc_auc_score(y_val, val_probs)
        fold_scores.append(score)
        print(f"  Fold {fold + 1} AUC: {score:.5f}")

        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    overall = roc_auc_score(y, oof_preds)
    print(f"\n  Seed {seed} OOF AUC: {overall:.5f}")
    return oof_preds, test_preds, overall

def main():
    print(f"{'='*80}")
    print(f"S6E2_{CFG.VERSION}_RealMLP_MultiSeed (Kaggle Env)")
    print(f"Seeds: {CFG.SEEDS}")
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

    all_oof = {}
    all_test = {}
    all_scores = {}

    for seed in CFG.SEEDS:
        oof_preds, test_preds, score = train_single_seed(seed, X, y, X_test, CFG.N_FOLDS)
        all_oof[seed] = oof_preds
        all_test[seed] = test_preds
        all_scores[seed] = score

    print(f"\n{'='*70}")
    print(f"MULTI-SEED ENSEMBLE RESULTS")
    print(f"{'='*70}")

    seeds = list(all_oof.keys())

    # Simple average
    avg_oof = np.mean([all_oof[s] for s in seeds], axis=0)
    avg_test = np.mean([all_test[s] for s in seeds], axis=0)
    avg_score = roc_auc_score(y, avg_oof)
    print(f"Simple Average ({len(seeds)} seeds): {avg_score:.5f}")

    # Rank average
    rank_oof = np.mean([rankdata(all_oof[s]) / len(y) for s in seeds], axis=0)
    rank_test = np.mean([rankdata(all_test[s]) / len(X_test) for s in seeds], axis=0)
    rank_score = roc_auc_score(y, rank_oof)
    print(f"Rank Average: {rank_score:.5f}")

    # Save best (simple average)
    pd.DataFrame({'id': train['id'], 'Heart Disease_prob': avg_oof}).to_csv("oof_v48.csv", index=False)
    pd.DataFrame({'id': test['id'], 'Heart Disease': avg_test}).to_csv("submission.csv", index=False)

    # Save individual seeds for blending
    for seed in CFG.SEEDS:
        pd.DataFrame({'id': train['id'], 'Heart Disease_prob': all_oof[seed]}).to_csv(f"oof_v48_seed{seed}.csv", index=False)
        pd.DataFrame({'id': test['id'], 'Heart Disease': all_test[seed]}).to_csv(f"submission_v48_seed{seed}.csv", index=False)

    elapsed = (time.time() - start_time) / 60
    print(f"\nSaved: submission.csv, oof_v48.csv + individual seed files")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
