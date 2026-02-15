"""V17 CatBoost Deotte Clone - Kaggle Environment."""
import warnings
try:
    import cudf.pandas
    cudf.pandas.install()
    print("cuDF loaded!")
except Exception:
    print("cuDF not available, using pandas.")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time
import os
import gc

warnings.filterwarnings('ignore')

class CFG:
    VERSION = "V17"
    CAT_PARAMS = {
        'iterations': 50000,
        'learning_rate': 0.0025,
        'depth': 3,
        'subsample': 0.8,
        'random_seed': 42,
        'early_stopping_rounds': 1000,
        'eval_metric': 'AUC',
        'task_type': 'GPU',
        'bootstrap_type': 'Bernoulli',
        'allow_writing_files': False
    }
    SEED = 42
    N_FOLDS = 15
    INNER_FOLDS = 15
    TRAIN_PATH = '/kaggle/input/competitions/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/competitions/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/datasets/rishidamarla/heart-disease-prediction/Heart_Disease_Prediction.csv'

def main():
    print(f"{'='*80}")
    print(f"S6E2_{CFG.VERSION}_CatBoost_Deotte (Kaggle Env)")
    print(f"{'='*80}")
    start_time = time.time()

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

    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']

    NEW_NUMS = []
    NUM_AS_CAT = []

    # Frequency Encoding
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{cat}')

    # Numerical as Categorical
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test, orig]:
            df[_new_col] = df[col].astype(str).astype('category')

    FEATURES = NUMS + CATS + NEW_NUMS + NUM_AS_CAT
    STATS = ['mean']
    TE_COLUMNS = NUM_AS_CAT + CATS
    TO_REMOVE = NUM_AS_CAT + CATS

    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    roc_auc_folds = []

    X_orig = orig[FEATURES + ['Heart Disease']].copy()
    y_orig = orig['Heart Disease'].copy()

    print(f"\nStarting {CFG.N_FOLDS}-Fold CV with Inner Fold TE...")

    for i, (train_index, val_index) in enumerate(kf.split(train)):
        X_train = train.loc[train_index, FEATURES + ['Heart Disease']].reset_index(drop=True).copy()
        y_train = train.loc[train_index, 'Heart Disease']

        X_train = pd.concat([X_train, X_orig], axis=0).reset_index(drop=True).copy()
        y_train = pd.concat([y_train, y_orig], axis=0).reset_index(drop=True).copy()

        X_val = train.loc[val_index, FEATURES].reset_index(drop=True).copy()
        y_val = train.loc[val_index, 'Heart Disease']
        X_test = test[FEATURES].reset_index(drop=True).copy()

        kf2 = KFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)

        for j, (train_index2, val_index2) in enumerate(kf2.split(X_train)):
            X_train2 = X_train.loc[train_index2, FEATURES + ['Heart Disease']].copy()
            X_val2 = X_train.loc[val_index2, FEATURES].copy()

            for col in TE_COLUMNS:
                tmp = X_train2.groupby(col)['Heart Disease'].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                X_val2 = X_val2.merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_train.loc[val_index2, c] = X_val2[c].values.astype("float32")

        for col in TE_COLUMNS:
            tmp = X_train.groupby(col)['Heart Disease'].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            X_val = X_val.merge(tmp, on=col, how="left")
            X_test = X_test.merge(tmp, on=col, how="left")

        for df in [X_train, X_val, X_test]:
            drop_cols = [c for c in TO_REMOVE if c in df.columns]
            df.drop(columns=drop_cols, inplace=True)

        if 'Heart Disease' in X_train.columns:
            X_train = X_train.drop(['Heart Disease'], axis=1)

        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        model = CatBoostClassifier(**CFG.CAT_PARAMS)
        model.fit(train_pool, eval_set=val_pool, verbose=False, use_best_model=True)

        val_p = model.predict_proba(X_val)[:, 1]
        oof[val_index] = val_p

        roc_auc_fold = roc_auc_score(y_val, val_p)
        roc_auc_folds.append(roc_auc_fold)
        print(f"Fold {i+1} AUC: {roc_auc_fold:.5f}")

        pred += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS

        del X_train, X_val, X_test, model, train_pool, val_pool
        gc.collect()

    overall_score = roc_auc_score(train['Heart Disease'], oof)
    print(f"\nOverall CV AUC: {overall_score:.5f}")

    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': pred})
    sub.to_csv("submission.csv", index=False)

    oof_df = pd.DataFrame({'id': train['id'].values, 'target': train['Heart Disease'].values, 'pred': oof})
    oof_df.to_csv("oof_v17.csv", index=False)

    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: submission.csv, oof_v17.csv")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
