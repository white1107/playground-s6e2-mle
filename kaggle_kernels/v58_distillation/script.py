"""V58 Distillation: Train RealMLP on pseudo-labels from CatBoost+RealMLP blend.

Steps:
1. Train CatBoost Ordered (V39-style) -> test predictions
2. Train RealMLP (V40-style) -> test predictions
3. Blend: 65% RealMLP + 35% CatBoost
4. Pseudo-label confident predictions (>0.99 positive, <0.01 negative)
5. Retrain RealMLP on train + pseudo-labeled test data
6. Output distilled submission
"""
import subprocess
subprocess.check_call(['pip', 'install', 'pytabkit', '-q'])

import warnings
import numpy as np
import pandas as pd
import torch
import re
import gc
import time
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, RobustScaler
from pytabkit import RealMLP_TD_Classifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

TRAIN_PATH = '/kaggle/input/competitions/playground-series-s6e2/train.csv'
TEST_PATH = '/kaggle/input/competitions/playground-series-s6e2/test.csv'
ORIG_PATH = '/kaggle/input/datasets/rishidamarla/heart-disease-prediction/Heart_Disease_Prediction.csv'

REALMLP_PARAMS = {
    'device': DEVICE, 'random_state': 42, 'verbosity': 2,
    'n_epochs': 100, 'batch_size': 256, 'n_ens': 8,
    'use_early_stopping': True, 'early_stopping_additive_patience': 20,
    'early_stopping_multiplicative_patience': 1, 'act': "mish",
    'embedding_size': 8, 'first_layer_lr_factor': 0.5962121993798933,
    'hidden_sizes': "rectangular", 'hidden_width': 384, 'lr': 0.04,
    'ls_eps': 0.011498317194338772, 'ls_eps_sched': "coslog4",
    'max_one_hot_cat_size': 18, 'n_hidden_layers': 4,
    'p_drop': 0.07301419697186451, 'p_drop_sched': "flat_cos",
    'plr_hidden_1': 16, 'plr_hidden_2': 8,
    'plr_lr_factor': 0.1151437622270563, 'plr_sigma': 2.3316811282666916,
    'scale_lr_factor': 2.244801835541429,
    'sq_mom': 1.0 - 0.011834054955582318, 'wd': 0.02369230879235962,
}

def add_orig_features(df, original, base_features):
    df_temp = df.copy()
    for col in base_features:
        if col in original.columns:
            stats = original.groupby(col)['Heart Disease'].agg(['mean', 'median', 'std', 'skew', 'count']).reset_index()
            stats.columns = [col] + [f"orig_{col}_{s}" for s in ['mean', 'median', 'std', 'skew', 'count']]
            df_temp = df_temp.merge(stats, on=col, how='left')
            fill_values = {
                f"orig_{col}_mean": original['Heart Disease'].mean(),
                f"orig_{col}_median": original['Heart Disease'].median(),
                f"orig_{col}_std": 0, f"orig_{col}_skew": 0, f"orig_{col}_count": 0
            }
            df_temp = df_temp.fillna(value=fill_values)
    return df_temp

def main():
    print("=" * 80)
    print("S6E2 V58: Distillation (CatBoost+RealMLP -> Pseudo-Label -> RealMLP)")
    print("=" * 80)
    start = time.time()

    # Load data
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    original = pd.read_csv(ORIG_PATH)

    print(f"Train: {train_raw.shape}, Test: {test_raw.shape}, Orig: {original.shape}")

    le = LabelEncoder()
    train_raw['Heart Disease'] = le.fit_transform(train_raw['Heart Disease'])
    original['Heart Disease'] = le.fit_transform(original['Heart Disease'])

    # ========== STEP 1: Train CatBoost (V39-style) ==========
    print("\n" + "=" * 60)
    print("STEP 1: Training CatBoost Ordered (teacher)")
    print("=" * 60)

    norm = lambda x: re.sub(r"[^\w\s]", "", x.strip().lower()).replace(" ", "_")
    train_cb = train_raw.copy()
    test_cb = test_raw.copy()
    train_cb.columns = [norm(c) for c in train_cb.columns]
    test_cb.columns = [norm(c) for c in test_cb.columns]

    target_col = 'heart_disease'
    cat_cols_raw = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results',
                    'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
    num_cols_raw = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']

    global_mean = train_cb[target_col].mean()
    stats_mean, stats_count = {}, {}
    for col in num_cols_raw + cat_cols_raw:
        cn = norm(col)
        stats_mean[col] = train_cb.groupby(cn)[target_col].mean().to_dict()
        stats_count[col] = train_cb.groupby(cn)[target_col].count().to_dict()

    # FE for CatBoost
    for col in num_cols_raw + cat_cols_raw:
        cn = norm(col)
        for df in [train_cb, test_cb]:
            df[f'mean_{cn}'] = df[cn].map(stats_mean.get(col, {})).fillna(global_mean)
            df[f'count_{cn}'] = df[cn].map(stats_count.get(col, {})).fillna(0)
        freq = train_cb[cn].value_counts(normalize=True).to_dict()
        for df in [train_cb, test_cb]:
            df[f'freq_{cn}'] = df[cn].map(freq).fillna(0)

    bin_targets = [norm(c) for c in num_cols_raw]
    kbd = KBinsDiscretizer(n_bins=10, strategy='uniform', encode='ordinal')
    train_cb[[f'bin_{c}' for c in bin_targets]] = kbd.fit_transform(train_cb[bin_targets]).astype(int)
    test_cb[[f'bin_{c}' for c in bin_targets]] = kbd.transform(test_cb[bin_targets]).astype(int)
    rs = RobustScaler()
    train_cb[bin_targets] = rs.fit_transform(train_cb[bin_targets])
    test_cb[bin_targets] = rs.transform(test_cb[bin_targets])

    cat_cols_norm = [norm(c) for c in cat_cols_raw]
    ordinal_cols = [f'bin_{c}' for c in [norm(c) for c in num_cols_raw]] + cat_cols_norm
    for df in [train_cb, test_cb]:
        for c in ordinal_cols:
            df[c] = df[c].astype(str).astype('category')

    features_cb = [c for c in train_cb.columns if c not in ['id', target_col]]
    X_cb = train_cb[features_cb]
    y_cb = train_cb[target_col]
    X_test_cb = test_cb[features_cb]

    cb_test_preds = np.zeros(len(test_raw))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_cb, y_cb)):
        model = CatBoostClassifier(
            iterations=8000, learning_rate=0.015, depth=5, l2_leaf_reg=5.0,
            random_strength=1.5, boosting_type='Ordered', bootstrap_type='Bernoulli',
            subsample=0.8, eval_metric='AUC', auto_class_weights='Balanced',
            early_stopping_rounds=200, task_type='GPU', random_seed=42, verbose=0
        )
        model.fit(X_cb.iloc[tr_idx], y_cb.iloc[tr_idx],
                  eval_set=[(X_cb.iloc[va_idx], y_cb.iloc[va_idx])],
                  cat_features=ordinal_cols, use_best_model=True)
        cb_test_preds += model.predict_proba(X_test_cb)[:, 1] / 5
        print(f"  CatBoost Fold {fold+1} done")
        del model; gc.collect()

    # ========== STEP 2: Train RealMLP (V40-style) ==========
    print("\n" + "=" * 60)
    print("STEP 2: Training RealMLP (teacher)")
    print("=" * 60)

    base_features = [c for c in train_raw.columns if c not in ['Heart Disease', 'id']]
    train_mlp = add_orig_features(train_raw, original, base_features)
    test_mlp = add_orig_features(test_raw, original, base_features)

    X_mlp = train_mlp.drop(['id', 'Heart Disease'], axis=1)
    y_mlp = train_mlp['Heart Disease']
    X_test_mlp = test_mlp.drop(['id'], axis=1)

    for col in X_mlp.columns:
        X_mlp[col] = X_mlp[col].astype(str).astype('category')
        X_test_mlp[col] = X_test_mlp[col].astype(str).astype('category')

    mlp_test_preds = np.zeros(len(test_raw))
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf2.split(X_mlp, y_mlp)):
        model = RealMLP_TD_Classifier(**REALMLP_PARAMS)
        model.fit(X_mlp.iloc[tr_idx], y_mlp.iloc[tr_idx].values,
                  X_mlp.iloc[va_idx], y_mlp.iloc[va_idx].values)
        mlp_test_preds += model.predict_proba(X_test_mlp)[:, 1] / 5
        print(f"  RealMLP Fold {fold+1} done")
        if DEVICE == 'cuda': torch.cuda.empty_cache()

    # ========== STEP 3: Blend & Pseudo-label ==========
    print("\n" + "=" * 60)
    print("STEP 3: Blending & Pseudo-labeling")
    print("=" * 60)

    blend = 0.65 * mlp_test_preds + 0.35 * cb_test_preds
    print(f"Blend range: {blend.min():.4f} - {blend.max():.4f}")

    # Pseudo-label
    high_conf = blend > 0.99
    low_conf = blend < 0.01
    n_pos = high_conf.sum()
    n_neg = low_conf.sum()
    print(f"Pseudo-labels: {n_pos} positive, {n_neg} negative ({n_pos+n_neg} total)")

    # Create pseudo-labeled test data
    pl_mask = high_conf | low_conf
    if pl_mask.sum() > 0:
        test_pl = test_raw[pl_mask].copy()
        test_pl['Heart Disease'] = np.where(blend[pl_mask] > 0.5, 1, 0)
        train_aug = pd.concat([train_raw, test_pl], axis=0).reset_index(drop=True)
        print(f"Augmented train: {train_aug.shape} (original {train_raw.shape[0]} + {len(test_pl)} PL)")
    else:
        train_aug = train_raw.copy()
        print("No confident pseudo-labels found!")

    # ========== STEP 4: Retrain RealMLP on augmented data ==========
    print("\n" + "=" * 60)
    print("STEP 4: Retraining RealMLP (distilled student)")
    print("=" * 60)

    train_aug2 = add_orig_features(train_aug, original, base_features)
    test_mlp2 = add_orig_features(test_raw, original, base_features)

    X_aug = train_aug2.drop(['id', 'Heart Disease'], axis=1)
    y_aug = train_aug2['Heart Disease']
    X_test2 = test_mlp2.drop(['id'], axis=1)

    for col in X_aug.columns:
        X_aug[col] = X_aug[col].astype(str).astype('category')
        X_test2[col] = X_test2[col].astype(str).astype('category')

    # Only validate on original train data
    n_orig = len(train_raw)
    X_orig = X_aug.iloc[:n_orig]
    y_orig = y_aug.iloc[:n_orig]

    distilled_test = np.zeros(len(test_raw))
    oof_preds = np.zeros(n_orig)
    fold_scores = []

    skf3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf3.split(X_orig, y_orig)):
        # Train on original fold train + ALL pseudo-labels
        X_tr = pd.concat([X_orig.iloc[tr_idx], X_aug.iloc[n_orig:]])
        y_tr = pd.concat([y_orig.iloc[tr_idx], y_aug.iloc[n_orig:]])
        X_val = X_orig.iloc[va_idx]
        y_val = y_orig.iloc[va_idx]

        params = REALMLP_PARAMS.copy()
        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y_tr.values, X_val, y_val.values)

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[va_idx] = val_probs
        distilled_test += model.predict_proba(X_test2)[:, 1] / 5

        score = roc_auc_score(y_val, val_probs)
        fold_scores.append(score)
        print(f"  Distilled Fold {fold+1} AUC: {score:.5f}")
        if DEVICE == 'cuda': torch.cuda.empty_cache()

    oof_auc = roc_auc_score(y_orig, oof_preds)
    print(f"\nDistilled OOF AUC: {oof_auc:.5f}")
    print(f"Mean Fold: {np.mean(fold_scores):.5f}")

    # Also save the raw blend as alternative
    pd.DataFrame({'id': test_raw['id'], 'Heart Disease': distilled_test}).to_csv("submission.csv", index=False)
    pd.DataFrame({'id': test_raw['id'], 'Heart Disease': blend}).to_csv("submission_blend.csv", index=False)
    pd.DataFrame({'id': train_raw['id'], 'oof_pred': oof_preds}).to_csv("oof_v58.csv", index=False)

    elapsed = (time.time() - start) / 60
    print(f"\nSaved: submission.csv (distilled), submission_blend.csv (teacher blend)")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
