"""V35 XGB Tuned Deotte - Running on Kaggle Environment."""

# 1. Import RAPIDS (Must be first)
import warnings
try:
    import cudf.pandas
    cudf.pandas.install()
    print("cuDF (pandas accelerator) loaded successfully!")
except ImportError:
    print("cuDF not found. Falling back to standard pandas.")
except Exception as e:
    print(f"cuDF failed to load: {e}")
    print("Falling back to standard pandas.")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time
import os
import gc

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V35"
    DESCRIPTION = "XGB_Tuned_Deotte"

    XGB_PARAMS = {
        'n_estimators': 50000,
        'learning_rate': 0.0025,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'reg_lambda': 2.5,
        'reg_alpha': 0.1,
        'random_state': 42,
        'early_stopping_rounds': 1000,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'enable_categorical': True,
        'device': 'cuda',
        'tree_method': 'hist'
    }

    SEED = 42
    N_FOLDS = 15
    INNER_FOLDS = 15

    TRAIN_PATH = '/kaggle/input/competitions/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/competitions/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/datasets/rishidamarla/heart-disease-prediction/Heart_Disease_Prediction.csv'

    SUBMISSION_PATH = "submission.csv"
    OOF_PATH = "oof_v35.csv"

def main():
    print(f"{'='*80}")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"Running on Kaggle Environment")
    print(f"{'='*80}")
    start_time = time.time()

    # Debug: show available input directories
    import subprocess
    print("\n--- Available input directories ---")
    result = subprocess.run(['ls', '-la', '/kaggle/input/'], capture_output=True, text=True)
    print(result.stdout)
    for d in os.listdir('/kaggle/input/'):
        sub = os.path.join('/kaggle/input/', d)
        if os.path.isdir(sub):
            print(f"  {d}/: {os.listdir(sub)[:5]}")
    print("--- End debug ---\n")

    # 1. Load Data
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    try:
        orig = pd.read_csv(CFG.ORIG_PATH)
    except:
        orig = pd.DataFrame(columns=train.columns)

    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]
    orig.columns = [c.strip() for c in orig.columns]

    if train['Heart Disease'].dtype == 'object':
        train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    if len(orig) > 0 and orig['Heart Disease'].dtype == 'object':
        orig['Heart Disease'] = orig['Heart Disease'].map({'Absence': 0, 'Presence': 1})

    print(f"Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")

    # 2. Feature Engineering Setup
    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']

    NEW_NUMS = []
    NEW_CATS = []
    NUM_AS_CAT = []
    TO_REMOVE = []

    print("Applying Feature Engineering...")

    # Frequency Encoding
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat] if len(orig) > 0 else pd.Series(), test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            if len(df) > 0:
                df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{cat}')

    # Numerical as Categorical
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test, orig]:
            if len(df) > 0:
                df[_new_col] = df[col].astype(str).astype('category')

    FEATURES = NUMS + CATS + NEW_NUMS + NEW_CATS + NUM_AS_CAT
    STATS = ['mean']
    TE_COLUMNS = NUM_AS_CAT + CATS + NEW_CATS
    TO_REMOVE += NUM_AS_CAT + CATS + NEW_CATS

    # 3. Validation Loop
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

    oof = np.zeros((len(train)))
    pred = np.zeros((len(test)))
    roc_auc_folds = []

    X_orig = orig[FEATURES+['Heart Disease']].copy()
    y_orig = orig['Heart Disease'].copy()

    print(f"\nStarting {CFG.N_FOLDS}-Fold CV with Inner Fold TE...")

    for i, (train_index, val_index) in enumerate(kf.split(train)):

        # Outer Split
        X_train = train.loc[train_index, FEATURES+['Heart Disease']].reset_index(drop=True).copy()
        y_train = train.loc[train_index, 'Heart Disease']

        # Augment
        if len(orig) > 0:
            X_train = pd.concat([X_train, X_orig], axis=0).reset_index(drop=True).copy()
            y_train = pd.concat([y_train, y_orig], axis=0).reset_index(drop=True).copy()

        X_val = train.loc[val_index, FEATURES].reset_index(drop=True).copy()
        y_val = train.loc[val_index, 'Heart Disease']
        X_test = test[FEATURES].reset_index(drop=True).copy()

        # Inner CV for TE
        kf2 = KFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)

        for j, (train_index2, val_index2) in enumerate(kf2.split(X_train)):
            X_train2 = X_train.loc[train_index2, FEATURES + ['Heart Disease']].copy()
            X_val2   = X_train.loc[val_index2, FEATURES].copy()

            for col in TE_COLUMNS:
                tmp = X_train2.groupby(col)['Heart Disease'].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                X_val2 = X_val2.merge(tmp, on=col, how="left")

                for c in tmp.columns:
                    X_train.loc[val_index2, c] = X_val2[c].values.astype("float32")

        # Outer TE
        for col in TE_COLUMNS:
            tmp = X_train.groupby(col)['Heart Disease'].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")

            X_val = X_val.merge(tmp, on=col, how="left")
            X_test = X_test.merge(tmp, on=col, how="left")

        # Final Prep
        for df in [X_train, X_val, X_test]:
            cols = CATS + NEW_CATS + NUM_AS_CAT
            valid_cols = [c for c in cols if c in df.columns]
            if valid_cols:
                df[valid_cols] = df[valid_cols].astype(str).astype("category")

        # Drop columns
        drop_cols_train = [c for c in TO_REMOVE if c in X_train.columns]
        X_train.drop(columns=drop_cols_train, inplace=True)

        drop_cols_val = [c for c in TO_REMOVE if c in X_val.columns]
        X_val.drop(columns=drop_cols_val, inplace=True)

        drop_cols_test = [c for c in TO_REMOVE if c in X_test.columns]
        X_test.drop(columns=drop_cols_test, inplace=True)

        if 'Heart Disease' in X_train.columns:
            X_train = X_train.drop(['Heart Disease'], axis=1)

        # Train
        model = xgb.XGBClassifier(**CFG.XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        val_p = model.predict_proba(X_val)[:,1]
        oof[val_index] = val_p

        roc_auc_fold = roc_auc_score(y_val, val_p)
        roc_auc_folds.append(roc_auc_fold)
        print(f"Fold {i+1} AUC: {roc_auc_fold:.5f}")

        pred += model.predict_proba(X_test)[:,1] / CFG.N_FOLDS

        del X_train, X_val, X_test, model
        gc.collect()

    overall_score = roc_auc_score(train['Heart Disease'], oof)
    print(f"\nOverall CV AUC: {overall_score:.5f}")
    print(f"Mean Fold AUC: {np.mean(roc_auc_folds):.5f}")

    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': pred})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)

    oof_df = pd.DataFrame({'id': train['id'].values, 'target': train['Heart Disease'].values, 'pred': oof})
    oof_df.to_csv(CFG.OOF_PATH, index=False)

    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
