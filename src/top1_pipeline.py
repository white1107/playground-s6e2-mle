"""Top 1% Pipeline: Multi-Seed GBDT Trinity + LogisticRegression Stacking.

Implements the exact strategy from the Top 1% public notebook:
- Expert domain features + target encoding + frequency encoding
- Fixed hyperparameters (no Optuna - less overfitting)
- Multi-seed averaging for CatBoost, XGBoost, LightGBM (5 seeds each)
- Simple LogisticRegression stacking with only 3 raw predictions

Usage:
    python -m src.top1_pipeline
    python -m src.top1_pipeline --seeds 10
"""

import argparse
import gc
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# =============================================================================
# Config
# =============================================================================
N_SPLITS = 5
SEEDS = [42, 2024, 2025, 1234, 5678]
TARGET = "Heart Disease"
ID_COL = "id"

CAT_COLS = [
    "Sex", "Chest pain type", "EKG results", "Exercise angina",
    "Slope of ST", "Number of vessels fluro", "Thallium", "FBS over 120",
]


# =============================================================================
# Feature Engineering (matching Top 1% notebook)
# =============================================================================
def add_expert_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Cardiology
    df["rate_pressure_product"] = df["Max HR"] * df["BP"] / 1000
    theoretical_max = 220 - df["Age"]
    df["cardiac_reserve"] = (df["Max HR"] / theoretical_max).clip(0.5, 1.2)

    # Epidemiology
    df["risk_age"] = (df["Age"] > 55).astype(int)
    df["risk_male"] = (df["Sex"] == 1).astype(int)
    df["hypertension"] = (df["BP"] > 140).astype(int)
    df["high_chol"] = (df["Cholesterol"] > 200).astype(int)
    df["very_high_chol"] = (df["Cholesterol"] > 240).astype(int)
    df["score_proxy"] = (df["Age"] / 10).astype(int) + df["risk_male"] + df["very_high_chol"]

    # Metabolic
    df["metabolic_syndrome"] = df["hypertension"] + df["high_chol"] + df["FBS over 120"]

    # Exercise Physiology
    df["st_ratio"] = df["ST depression"] / (df["Max HR"] + 1)
    df["exercise_angina_x_st"] = df["Exercise angina"] * df["ST depression"]

    # Composite
    df["risk_score"] = (
        df["risk_age"] + df["risk_male"] + df["hypertension"]
        + df["high_chol"] + df["FBS over 120"]
    )
    df["severe_vessels"] = (df["Number of vessels fluro"] >= 2).astype(int)
    df["thallium_defect"] = (df["Thallium"] >= 6).astype(int)

    # Interactions
    df["age_x_vessels"] = df["Age"] * df["Number of vessels fluro"]
    df["rpp_x_st"] = df["rate_pressure_product"] * df["ST depression"]
    df["chol_x_bp"] = df["Cholesterol"] * df["BP"] / 10000

    return df


def target_encode_oof(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    target: np.ndarray,
    cat_cols: list[str],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOF-safe target encoding + frequency encoding."""
    tr_enc, te_enc = tr.copy(), te.copy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    glob_mean = target.mean()

    for col in cat_cols:
        tr_enc[f"{col}_TE"] = np.nan

        for tr_idx, va_idx in skf.split(tr_enc, target):
            mean_map = tr_enc.iloc[tr_idx].groupby(col)[TARGET].mean()
            tr_enc.loc[tr_enc.index[va_idx], f"{col}_TE"] = (
                tr_enc.iloc[va_idx][col].map(mean_map)
            )

        tr_enc[f"{col}_TE"] = tr_enc[f"{col}_TE"].fillna(glob_mean)
        full_map = tr_enc.groupby(col)[TARGET].mean()
        te_enc[f"{col}_TE"] = te_enc[col].map(full_map).fillna(glob_mean)

        # Frequency encoding
        freq_map = tr_enc[col].value_counts(normalize=True)
        tr_enc[f"{col}_freq"] = tr_enc[col].map(freq_map)
        te_enc[f"{col}_freq"] = te_enc[col].map(freq_map).fillna(0)

    return tr_enc, te_enc


# =============================================================================
# Model Training Functions (fixed hyperparameters)
# =============================================================================
def train_catboost_seed(X, y, X_test, seed, use_gpu=True):
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    cat_idx = [X.columns.get_loc(c) for c in CAT_COLS if c in X.columns]

    for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
        tr_p = Pool(X.iloc[tr_i], y[tr_i], cat_features=cat_idx)
        va_p = Pool(X.iloc[va_i], y[va_i], cat_features=cat_idx)
        model = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.05,
            depth=7,
            task_type="GPU" if use_gpu else "CPU",
            devices="0",
            verbose=0,
            early_stopping_rounds=100,
            random_seed=seed + fold,
        )
        model.fit(tr_p, eval_set=va_p)
        oof[va_i] = model.predict_proba(X.iloc[va_i])[:, 1]
        pred += model.predict_proba(X_test)[:, 1] / N_SPLITS

    return oof, pred


def train_xgb_seed(X, y, X_test, seed, use_gpu=True):
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
        model = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            early_stopping_rounds=100,
            random_state=seed + fold,
            verbosity=0,
        )
        model.fit(
            X.iloc[tr_i], y[tr_i],
            eval_set=[(X.iloc[va_i], y[va_i])],
            verbose=False,
        )
        oof[va_i] = model.predict_proba(X.iloc[va_i])[:, 1]
        pred += model.predict_proba(X_test)[:, 1] / N_SPLITS

    return oof, pred


def train_lgb_seed(X, y, X_test, seed):
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
        model = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.05,
            max_depth=7,
            random_state=seed + fold,
            verbose=-1,
        )
        model.fit(
            X.iloc[tr_i], y[tr_i],
            eval_set=[(X.iloc[va_i], y[va_i])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va_i] = model.predict_proba(X.iloc[va_i])[:, 1]
        pred += model.predict_proba(X_test)[:, 1] / N_SPLITS

    return oof, pred


# =============================================================================
# Main Pipeline
# =============================================================================
def run_top1_pipeline(n_seeds: int = 5, use_gpu: bool = True):
    print("=" * 60)
    print("Top 1% Pipeline: Multi-Seed GBDT Trinity")
    print(f"Seeds: {n_seeds} | Folds: {N_SPLITS} | GPU: {use_gpu}")
    print("=" * 60)

    # Load data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    label_map = {"Presence": 1, "Absence": 0}
    if train[TARGET].dtype == object:
        train[TARGET] = train[TARGET].map(label_map)
    y = train[TARGET].values.astype(np.float32)

    print(f"Train: {train.shape}, Test: {test.shape}")
    print(f"Positive rate: {y.mean():.2%}")

    # Feature engineering
    train_fe = add_expert_features(train)
    test_fe = add_expert_features(test)
    n_expert = train_fe.shape[1] - train.shape[1]
    print(f"Expert features added: {n_expert}")

    # Target encoding
    train_fe, test_fe = target_encode_oof(train_fe, test_fe, y, CAT_COLS)
    feature_cols = [c for c in train_fe.columns if c not in [ID_COL, TARGET]]
    print(f"Total features: {len(feature_cols)}")

    X = train_fe[feature_cols].copy()
    X_test = test_fe[feature_cols].copy()
    X_test = X_test[X.columns]

    # CatBoost requires cat features as int or str, not float
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype(int)
            X_test[col] = X_test[col].astype(int)

    seeds = SEEDS[:n_seeds]

    # Multi-seed training
    all_cb_oof, all_cb_pred = [], []
    all_xgb_oof, all_xgb_pred = [], []
    all_lgb_oof, all_lgb_pred = [], []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*40}")
        print(f"SEED {i + 1}/{len(seeds)} (seed={seed})")
        print(f"{'='*40}")

        print("  Training CatBoost...")
        cb_oof, cb_pred = train_catboost_seed(X, y, X_test, seed, use_gpu)
        cb_auc = roc_auc_score(y, cb_oof)
        print(f"  CatBoost AUC: {cb_auc:.5f}")

        print("  Training XGBoost...")
        xgb_oof, xgb_pred = train_xgb_seed(X, y, X_test, seed, use_gpu)
        xgb_auc = roc_auc_score(y, xgb_oof)
        print(f"  XGBoost AUC: {xgb_auc:.5f}")

        print("  Training LightGBM...")
        lgb_oof, lgb_pred = train_lgb_seed(X, y, X_test, seed)
        lgb_auc = roc_auc_score(y, lgb_oof)
        print(f"  LightGBM AUC: {lgb_auc:.5f}")

        all_cb_oof.append(cb_oof)
        all_cb_pred.append(cb_pred)
        all_xgb_oof.append(xgb_oof)
        all_xgb_pred.append(xgb_pred)
        all_lgb_oof.append(lgb_oof)
        all_lgb_pred.append(lgb_pred)

        gc.collect()

    # Average across seeds
    cb_oof_avg = np.mean(all_cb_oof, axis=0)
    cb_pred_avg = np.mean(all_cb_pred, axis=0)
    xgb_oof_avg = np.mean(all_xgb_oof, axis=0)
    xgb_pred_avg = np.mean(all_xgb_pred, axis=0)
    lgb_oof_avg = np.mean(all_lgb_oof, axis=0)
    lgb_pred_avg = np.mean(all_lgb_pred, axis=0)

    cb_avg_auc = roc_auc_score(y, cb_oof_avg)
    xgb_avg_auc = roc_auc_score(y, xgb_oof_avg)
    lgb_avg_auc = roc_auc_score(y, lgb_oof_avg)

    print(f"\n{'='*60}")
    print("Multi-Seed Averaged Results")
    print(f"{'='*60}")
    print(f"CatBoost  ({len(seeds)}-seed avg): {cb_avg_auc:.5f}")
    print(f"XGBoost   ({len(seeds)}-seed avg): {xgb_avg_auc:.5f}")
    print(f"LightGBM  ({len(seeds)}-seed avg): {lgb_avg_auc:.5f}")

    # Simple probability average
    simple_avg = (cb_oof_avg + xgb_oof_avg + lgb_oof_avg) / 3
    simple_avg_auc = roc_auc_score(y, simple_avg)
    print(f"Simple Average:              {simple_avg_auc:.5f}")

    # LogisticRegression stacking
    S_train = np.vstack([cb_oof_avg, xgb_oof_avg, lgb_oof_avg]).T
    S_test = np.vstack([cb_pred_avg, xgb_pred_avg, lgb_pred_avg]).T

    meta = LogisticRegression(random_state=42, max_iter=1000)
    meta.fit(S_train, y)

    stacked_train_pred = meta.predict_proba(S_train)[:, 1]
    stacked_auc = roc_auc_score(y, stacked_train_pred)
    final_pred = meta.predict_proba(S_test)[:, 1]

    print(f"Stacked (LR):                {stacked_auc:.5f}")
    print(f"Meta weights: CB={meta.coef_[0][0]:.3f}, "
          f"XGB={meta.coef_[0][1]:.3f}, LGB={meta.coef_[0][2]:.3f}")

    # Save outputs
    os.makedirs("output/predictions", exist_ok=True)
    os.makedirs("output/submissions", exist_ok=True)

    # Save individual model OOFs and submissions
    for name, oof, pred in [
        ("cat_top1", cb_oof_avg, cb_pred_avg),
        ("xgb_top1", xgb_oof_avg, xgb_pred_avg),
        ("lgbm_top1", lgb_oof_avg, lgb_pred_avg),
    ]:
        pd.DataFrame({"id": train["id"], "oof_pred": oof}).to_csv(
            f"output/predictions/oof_{name}_eng.csv", index=False
        )
        pd.DataFrame({"id": test["id"], "Heart Disease": pred}).to_csv(
            f"output/submissions/submission_{name}_eng_kfold.csv", index=False
        )

    # Save stacked submission
    pd.DataFrame({"id": test["id"], "Heart Disease": final_pred}).to_csv(
        "output/submissions/submission_top1_stacked.csv", index=False
    )

    # Save simple average submission
    simple_avg_pred = (cb_pred_avg + xgb_pred_avg + lgb_pred_avg) / 3
    pd.DataFrame({"id": test["id"], "Heart Disease": simple_avg_pred}).to_csv(
        "output/submissions/submission_top1_avg.csv", index=False
    )

    # Save CatBoost-only submission (often best single model)
    pd.DataFrame({"id": test["id"], "Heart Disease": cb_pred_avg}).to_csv(
        "output/submissions/submission_top1_cat_only.csv", index=False
    )

    print(f"\n{'='*60}")
    print("Submissions saved:")
    print("  submission_top1_stacked.csv   (LR stacking)")
    print("  submission_top1_avg.csv       (simple average)")
    print("  submission_top1_cat_only.csv  (CatBoost only)")
    print(f"{'='*60}")

    return {
        "cb_auc": cb_avg_auc,
        "xgb_auc": xgb_avg_auc,
        "lgb_auc": lgb_avg_auc,
        "simple_avg_auc": simple_avg_auc,
        "stacked_auc": stacked_auc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top 1% Pipeline")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds (default: 5)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU")
    args = parser.parse_args()

    run_top1_pipeline(n_seeds=args.seeds, use_gpu=not args.no_gpu)
