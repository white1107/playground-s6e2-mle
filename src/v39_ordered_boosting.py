"""V39: CatBoost Ordered Boosting with Global Stats + Freq + KBins.

Replicates the BlamerX V39 approach (LB 0.95390):
- CatBoost Ordered Boosting (depth=5, lr=0.015, 8000 iter)
- Global target statistics (mean, count per feature value)
- Frequency encoding on all features
- KBins discretization + RobustScaler on numericals
- auto_class_weights='Balanced'

Usage:
    python -m src.v39_ordered_boosting
"""

import gc
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, RobustScaler

warnings.filterwarnings("ignore")

# Config
N_FOLDS = 5
SEEDS = [42]
TARGET = "Heart Disease"

CAT_COLS = [
    "Sex", "Chest pain type", "FBS over 120", "EKG results",
    "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium",
]
NUM_COLS = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]

CB_PARAMS = {
    "iterations": 8000,
    "learning_rate": 0.015,
    "depth": 5,
    "l2_leaf_reg": 5.0,
    "random_strength": 1.5,
    "boosting_type": "Ordered",
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "early_stopping_rounds": 200,
    "task_type": "GPU",
    "devices": "0",
    "random_seed": 42,
    "verbose": 500,
}


def normalize_col(name):
    return re.sub(r"[^\w\s]", "", name.strip().lower()).replace(" ", "_")


def apply_feature_engineering(df, stats_mean, stats_count, global_mean,
                              num_cols, cat_cols, is_train=False,
                              _state={}):
    out = df.copy()

    # 1. Global Target Statistics
    for col in num_cols + cat_cols:
        col_n = normalize_col(col)
        out[f"mean_{col_n}"] = out[col_n].map(stats_mean.get(col, {})).fillna(global_mean)
        out[f"count_{col_n}"] = out[col_n].map(stats_count.get(col, {})).fillna(0)

    # 2. Frequency Encoding
    for col in num_cols + cat_cols:
        col_n = normalize_col(col)
        if is_train:
            freq = out[col_n].value_counts(normalize=True).to_dict()
            _state[f"freq_{col}"] = freq
        else:
            freq = _state.get(f"freq_{col}", {})
        out[f"freq_{col_n}"] = out[col_n].map(freq).fillna(0)

    # 3. KBins Discretization
    bin_targets = [normalize_col(c) for c in num_cols]
    if is_train:
        kbd = KBinsDiscretizer(n_bins=10, strategy="uniform", encode="ordinal")
        out[[f"bin_{c}" for c in bin_targets]] = kbd.fit_transform(out[bin_targets]).astype(int)
        _state["kbd"] = kbd
    else:
        out[[f"bin_{c}" for c in bin_targets]] = _state["kbd"].transform(out[bin_targets]).astype(int)

    # 4. RobustScaler
    if is_train:
        rs = RobustScaler()
        out[bin_targets] = rs.fit_transform(out[bin_targets])
        _state["rs"] = rs
    else:
        out[bin_targets] = _state["rs"].transform(out[bin_targets])

    return out


def main():
    print("=" * 60)
    print("V39: CatBoost Ordered Boosting + Global Stats")
    print("=" * 60)
    start = time.time()

    # Load
    train_raw = pd.read_csv("data/train.csv")
    test_raw = pd.read_csv("data/test.csv")

    # Normalize columns
    train = train_raw.copy()
    test = test_raw.copy()
    train.columns = [normalize_col(c) for c in train.columns]
    test.columns = [normalize_col(c) for c in test.columns]

    target_col = normalize_col(TARGET)
    if train[target_col].dtype == "object":
        train[target_col] = train[target_col].map({"Presence": 1, "Absence": 0})

    # Global statistics (computed on full train - leakage handled by Ordered Boosting)
    global_mean = train[target_col].mean()
    stats_mean = {}
    stats_count = {}
    for col in NUM_COLS + CAT_COLS:
        col_n = normalize_col(col)
        stats_mean[col] = train.groupby(col_n)[target_col].mean().to_dict()
        stats_count[col] = train.groupby(col_n)[target_col].count().to_dict()

    # Feature engineering
    fe_state = {}
    train_fe = apply_feature_engineering(
        train, stats_mean, stats_count, global_mean,
        NUM_COLS, CAT_COLS, is_train=True, _state=fe_state
    )
    test_fe = apply_feature_engineering(
        test, stats_mean, stats_count, global_mean,
        NUM_COLS, CAT_COLS, is_train=False, _state=fe_state
    )

    # Categorical features for CatBoost
    cat_cols_norm = [normalize_col(c) for c in CAT_COLS]
    num_cols_norm = [normalize_col(c) for c in NUM_COLS]
    ordinal_cols = [f"bin_{c}" for c in num_cols_norm] + cat_cols_norm

    for df in [train_fe, test_fe]:
        for c in ordinal_cols:
            df[c] = df[c].astype(str).astype("category")

    features = [c for c in train_fe.columns if c not in ["id", target_col]]
    print(f"Features: {len(features)}")

    X = train_fe[features]
    y = train_fe[target_col]
    X_test = test_fe[features]

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    for seed in SEEDS:
        print(f"\nSeed: {seed}")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
            params = CB_PARAMS.copy()
            params["random_seed"] = seed

            model = CatBoostClassifier(**params)
            model.fit(
                X.iloc[tr_i], y.iloc[tr_i],
                eval_set=(X.iloc[va_i], y.iloc[va_i]),
                cat_features=ordinal_cols,
                use_best_model=True,
            )

            val_pred = model.predict_proba(X.iloc[va_i])[:, 1]
            oof_preds[va_i] = val_pred
            test_preds += model.predict_proba(X_test)[:, 1] / (len(SEEDS) * N_FOLDS)

            auc = roc_auc_score(y.iloc[va_i], val_pred)
            print(f"  Fold {fold} AUC: {auc:.5f}")

            del model
            gc.collect()

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall OOF AUC: {oof_auc:.5f}")

    # Save
    os.makedirs("output/predictions", exist_ok=True)
    os.makedirs("output/submissions", exist_ok=True)

    pd.DataFrame({"id": train_raw["id"], "oof_pred": oof_preds}).to_csv(
        "output/predictions/oof_v39_ordered.csv", index=False
    )
    pd.DataFrame({"id": test_raw["id"], "Heart Disease": test_preds}).to_csv(
        "output/submissions/submission_v39_ordered.csv", index=False
    )

    elapsed = (time.time() - start) / 60
    print(f"Saved: submission_v39_ordered.csv, oof_v39_ordered.csv")
    print(f"Total Time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
