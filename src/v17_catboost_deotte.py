"""V17: CatBoost Deotte Clone (Inner-Fold TE + Freq + 15-fold).

Replicates the BlamerX V17 approach (LB 0.95385):
- CatBoost depth=3, lr=0.0025, 50000 iterations
- 15-fold CV with 15 inner folds for target encoding
- Frequency encoding on all features (train+test+orig)
- Numerical columns converted to categorical for TE
- Original data augmented into training folds
- Drops raw categoricals after TE (relies on TE features only)

Usage:
    python -m src.v17_catboost_deotte
"""

import gc
import os
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# Config
N_FOLDS = 15
INNER_FOLDS = 15
SEED = 42

CATS = ["Age", "Sex", "Chest pain type", "FBS over 120", "Exercise angina", "Thallium"]
NUMS = ["BP", "Cholesterol", "Max HR", "ST depression", "Slope of ST",
        "Number of vessels fluro", "EKG results"]

CB_PARAMS = {
    "iterations": 50000,
    "learning_rate": 0.0025,
    "depth": 3,
    "subsample": 0.8,
    "random_seed": 42,
    "early_stopping_rounds": 1000,
    "eval_metric": "AUC",
    "task_type": "CPU",
    # "devices": "0",  # CPU mode while V40 uses GPU
    "bootstrap_type": "Bernoulli",
    "allow_writing_files": False,
    "verbose": 0,
}


def main():
    print("=" * 60)
    print("V17: CatBoost Deotte Clone (Inner-Fold TE)")
    print(f"Folds: {N_FOLDS}, Inner Folds: {INNER_FOLDS}")
    print("=" * 60)
    start = time.time()

    # Load data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    try:
        orig = pd.read_csv("data/Heart_Disease_Prediction.csv")
    except FileNotFoundError:
        orig = pd.DataFrame(columns=train.columns)

    for df in [train, test, orig]:
        df.columns = [c.strip() for c in df.columns]

    # Map target
    if train["Heart Disease"].dtype == "object":
        train["Heart Disease"] = train["Heart Disease"].map({"Absence": 0, "Presence": 1})
    if len(orig) > 0 and orig["Heart Disease"].dtype == "object":
        orig["Heart Disease"] = orig["Heart Disease"].map({"Absence": 0, "Presence": 1})

    print(f"Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")

    # Feature Engineering
    NEW_NUMS = []
    NUM_AS_CAT = []
    TO_REMOVE = []

    # Frequency Encoding (on train+test+orig combined)
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")
        NEW_NUMS.append(f"FREQ_{col}")

    # Numerical as Categorical
    for col in NUMS:
        new_col = f"CAT_{col}"
        NUM_AS_CAT.append(new_col)
        for df in [train, test, orig]:
            df[new_col] = df[col].astype(str).astype("category")

    FEATURES = NUMS + CATS + NEW_NUMS + NUM_AS_CAT
    TE_COLUMNS = NUM_AS_CAT + CATS
    TO_REMOVE = NUM_AS_CAT + CATS  # Drop raw cats after TE

    print(f"Features before TE: {len(FEATURES)}")

    # CV Loop
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    fold_aucs = []

    X_orig = orig[FEATURES + ["Heart Disease"]].copy() if len(orig) > 0 else None
    y_orig = orig["Heart Disease"].copy() if len(orig) > 0 else None

    for i, (train_index, val_index) in enumerate(kf.split(train)):
        # Outer split
        X_train = train.loc[train_index, FEATURES + ["Heart Disease"]].reset_index(drop=True).copy()
        y_train = train.loc[train_index, "Heart Disease"]

        # Augment with original data
        if X_orig is not None and len(X_orig) > 0:
            X_train = pd.concat([X_train, X_orig], axis=0).reset_index(drop=True).copy()
            y_train = pd.concat([y_train, y_orig], axis=0).reset_index(drop=True).copy()

        X_val = train.loc[val_index, FEATURES].reset_index(drop=True).copy()
        y_val = train.loc[val_index, "Heart Disease"]
        X_test_fold = test[FEATURES].reset_index(drop=True).copy()

        # Inner CV for Target Encoding
        kf2 = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=42)

        for j, (tr_idx2, va_idx2) in enumerate(kf2.split(X_train)):
            X_train2 = X_train.loc[tr_idx2, FEATURES + ["Heart Disease"]].copy()
            X_val2 = X_train.loc[va_idx2, FEATURES].copy()

            for col in TE_COLUMNS:
                tmp = X_train2.groupby(col)["Heart Disease"].agg(["mean"])
                tmp.columns = [f"TE1_{col}_mean"]

                X_val2 = X_val2.merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_train.loc[va_idx2, c] = X_val2[c].values.astype("float32")

        # Outer TE (for val and test)
        for col in TE_COLUMNS:
            tmp = X_train.groupby(col)["Heart Disease"].agg(["mean"])
            tmp.columns = [f"TE1_{col}_mean"]
            tmp = tmp.astype("float32")

            X_val = X_val.merge(tmp, on=col, how="left")
            X_test_fold = X_test_fold.merge(tmp, on=col, how="left")

        # Drop raw categoricals (Deotte strategy: rely on TE only)
        for df in [X_train, X_val, X_test_fold]:
            drop_cols = [c for c in TO_REMOVE if c in df.columns]
            df.drop(columns=drop_cols, inplace=True)

        if "Heart Disease" in X_train.columns:
            X_train = X_train.drop(["Heart Disease"], axis=1)

        # Train CatBoost
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(train_pool, eval_set=val_pool, verbose=False, use_best_model=True)

        val_p = model.predict_proba(X_val)[:, 1]
        oof[val_index] = val_p

        auc = roc_auc_score(y_val, val_p)
        fold_aucs.append(auc)
        print(f"Fold {i + 1}/{N_FOLDS} AUC: {auc:.5f}")

        pred += model.predict_proba(X_test_fold)[:, 1] / N_FOLDS

        del X_train, X_val, X_test_fold, model, train_pool, val_pool
        gc.collect()

    oof_auc = roc_auc_score(train["Heart Disease"], oof)
    print(f"\nOverall OOF AUC: {oof_auc:.5f}")
    print(f"Mean Fold: {np.mean(fold_aucs):.5f}")

    # Save
    os.makedirs("output/predictions", exist_ok=True)
    os.makedirs("output/submissions", exist_ok=True)

    pd.DataFrame({"id": train["id"], "oof_pred": oof}).to_csv(
        "output/predictions/oof_v17_deotte.csv", index=False
    )
    pd.DataFrame({"id": test["id"], "Heart Disease": pred}).to_csv(
        "output/submissions/submission_v17_deotte.csv", index=False
    )

    elapsed = (time.time() - start) / 60
    print(f"Saved: submission_v17_deotte.csv, oof_v17_deotte.csv")
    print(f"Total Time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
