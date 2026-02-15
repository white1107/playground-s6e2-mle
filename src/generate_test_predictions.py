"""Generate test predictions for all model types using K-fold averaging.

Usage:
    python -m src.generate_test_predictions --model realmlp
    python -m src.generate_test_predictions --model tabnet
    python -m src.generate_test_predictions --model ft_transformer
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from src.feature_engineering import (
    add_domain_features,
    add_original_statistics,
    get_domain_features,
    load_original_data,
)


def generate_realmlp_test(args):
    from pytabkit import RealMLP_TD_Classifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    original = load_original_data()

    le = LabelEncoder()
    train["Heart Disease"] = le.fit_transform(train["Heart Disease"])
    original["Heart Disease"] = le.fit_transform(original["Heart Disease"])

    base_features = [c for c in train.columns if c not in ["Heart Disease", "id"]]
    train = add_original_statistics(train, original, base_features)
    test = add_original_statistics(test, original, base_features)

    X = train.drop(["id", "Heart Disease"], axis=1)
    X_test = test.drop(["id"], axis=1)
    y = train["Heart Disease"]

    for col in X.columns:
        X[col] = X[col].astype(str).astype("category")
    for col in X_test.columns:
        X_test[col] = X_test[col].astype(str).astype("category")

    param_grid = {
        "device": device, "random_state": args.seed, "verbosity": 0,
        "n_epochs": 100, "batch_size": 256, "n_ens": 8,
        "use_early_stopping": True, "early_stopping_additive_patience": 20,
        "early_stopping_multiplicative_patience": 1, "act": "mish",
        "embedding_size": 8, "first_layer_lr_factor": 0.5962121993798933,
        "hidden_sizes": "rectangular", "hidden_width": 384, "lr": 0.04,
        "ls_eps": 0.011498317194338772, "ls_eps_sched": "coslog4",
        "max_one_hot_cat_size": 18, "n_hidden_layers": 4,
        "p_drop": 0.07301419697186451, "p_drop_sched": "flat_cos",
        "plr_hidden_1": 16, "plr_hidden_2": 8,
        "plr_lr_factor": 0.1151437622270563, "plr_sigma": 2.3316811282666916,
        "scale_lr_factor": 2.244801835541429,
        "sq_mom": 1.0 - 0.011834054955582318, "wd": 0.02369230879235962,
    }

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{args.folds}")
        model = RealMLP_TD_Classifier(**param_grid)
        model.fit(X.iloc[train_idx], y.iloc[train_idx].values,
                  X.iloc[val_idx], y.iloc[val_idx].values)
        test_preds += model.predict_proba(X_test)[:, 1] / args.folds
        if device == "cuda":
            torch.cuda.empty_cache()

    return test_preds


def generate_tabnet_test(args):
    from pytorch_tabnet.tab_model import TabNetClassifier

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    original = load_original_data()

    if train["Heart Disease"].dtype == object:
        train["Heart Disease"] = train["Heart Disease"].map({"Presence": 1, "Absence": 0})

    num_features = ["Age", "BP", "Cholesterol", "Max HR", "ST depression", "Number of vessels fluro"]
    cat_features = ["Sex", "Chest pain type", "FBS over 120", "EKG results",
                    "Exercise angina", "Slope of ST", "Thallium"]
    base_features = [c for c in train.columns if c not in ["Heart Disease", "id"]]

    train = add_original_statistics(train, original, base_features)
    train = add_domain_features(train)
    test = add_original_statistics(test, original, base_features)
    test = add_domain_features(test)
    domain_feats = get_domain_features()

    orig_feat_cols = [c for c in train.columns if c.startswith("orig_")]
    all_num_features = num_features + domain_feats + orig_feat_cols
    all_features = all_num_features + cat_features

    X = train[all_features].copy()
    X_test = test[all_features].copy()
    y = train["Heart Disease"].values

    cat_idxs, cat_dims = [], []
    for i, col in enumerate(all_features):
        if col in cat_features:
            le = LabelEncoder()
            combined = pd.concat([X[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X[col] = le.transform(X[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            cat_idxs.append(i)
            cat_dims.append(len(le.classes_))

    X_vals = X.values.astype(np.float32)
    X_test_vals = X_test.values.astype(np.float32)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_vals, y)):
        print(f"Fold {fold + 1}/{args.folds}")
        clf = TabNetClassifier(
            cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=4,
            n_d=32, n_a=32, n_steps=5, gamma=1.5, lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=0.02),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type="entmax", verbose=0,
        )
        clf.fit(
            X_train=X_vals[train_idx], y_train=y[train_idx],
            eval_set=[(X_vals[val_idx], y[val_idx])],
            eval_name=["valid"], eval_metric=["auc"],
            max_epochs=150, patience=25, batch_size=2048,
            virtual_batch_size=256, num_workers=0, drop_last=False,
        )
        test_preds += clf.predict_proba(X_test_vals)[:, 1] / args.folds

    return test_preds


def generate_ft_transformer_test(args):
    from rtdl_revisiting_models import FTTransformer
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    original = load_original_data()

    if train["Heart Disease"].dtype == object:
        train["Heart Disease"] = train["Heart Disease"].map({"Presence": 1, "Absence": 0})

    num_features = ["Age", "BP", "Cholesterol", "Max HR", "ST depression", "Number of vessels fluro"]
    cat_features = ["Sex", "Chest pain type", "FBS over 120", "EKG results",
                    "Exercise angina", "Slope of ST", "Thallium"]
    base_features = [c for c in train.columns if c not in ["Heart Disease", "id"]]

    train = add_original_statistics(train, original, base_features)
    train = add_domain_features(train)
    test = add_original_statistics(test, original, base_features)
    test = add_domain_features(test)
    domain_feats = get_domain_features()

    orig_feat_cols = [c for c in train.columns if c.startswith("orig_")]
    all_num_features = num_features + domain_feats + orig_feat_cols

    y = train["Heart Disease"].values

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    combined_cat = pd.concat([train[cat_features].astype(str), test[cat_features].astype(str)])
    oe.fit(combined_cat)
    X_cat_train = oe.transform(train[cat_features].astype(str)).astype(int)
    X_cat_test = oe.transform(test[cat_features].astype(str)).astype(int)
    cat_cardinalities = [int(X_cat_train[:, i].max()) + 1 for i in range(X_cat_train.shape[1])]

    X_num_train = train[all_num_features].values.astype(np.float32)
    X_num_test = test[all_num_features].values.astype(np.float32)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    test_preds = np.zeros(len(test))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_num_train, y)):
        print(f"Fold {fold + 1}/{args.folds}")

        scaler = StandardScaler()
        x_num_tr = scaler.fit_transform(X_num_train[train_idx])
        x_num_va = scaler.transform(X_num_train[val_idx])
        x_num_te = scaler.transform(X_num_test)

        train_ds = TensorDataset(
            torch.tensor(x_num_tr, dtype=torch.float32),
            torch.tensor(X_cat_train[train_idx], dtype=torch.long),
            torch.tensor(y[train_idx], dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True)

        model = FTTransformer(
            n_cont_features=len(all_num_features),
            cat_cardinalities=cat_cardinalities, d_out=1,
            n_blocks=3, d_block=192, attention_n_heads=8,
            attention_dropout=0.2, ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4/3, ffn_dropout=0.1, residual_dropout=0.0,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)

        best_auc, best_state, patience_counter = 0.0, None, 0
        val_ds = TensorDataset(
            torch.tensor(x_num_va, dtype=torch.float32),
            torch.tensor(X_cat_train[val_idx], dtype=torch.long),
            torch.tensor(y[val_idx], dtype=torch.float32),
        )
        val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

        for epoch in range(80):
            model.train()
            for x_n, x_c, y_b in train_loader:
                x_n, x_c, y_b = x_n.to(device), x_c.to(device), y_b.to(device).unsqueeze(1)
                optimizer.zero_grad()
                loss = loss_fn(model(x_n, x_c), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            vp = []
            with torch.no_grad():
                for x_n, x_c, _ in val_loader:
                    vp.append(torch.sigmoid(model(x_n.to(device), x_c.to(device))).cpu().numpy().flatten())
            val_auc = roc_auc_score(y[val_idx], np.concatenate(vp))

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 12:
                    break

        model.load_state_dict(best_state)
        model.eval()
        # Batched inference for large test sets
        fold_test_parts = []
        batch_size_test = 4096
        with torch.no_grad():
            for start in range(0, len(x_num_te), batch_size_test):
                end = min(start + batch_size_test, len(x_num_te))
                x_n_b = torch.tensor(x_num_te[start:end], dtype=torch.float32).to(device)
                x_c_b = torch.tensor(X_cat_test[start:end], dtype=torch.long).to(device)
                fold_test_parts.append(torch.sigmoid(model(x_n_b, x_c_b)).cpu().numpy().flatten())
        fold_test = np.concatenate(fold_test_parts)
        test_preds += fold_test / args.folds
        print(f"  Fold {fold+1} best_val_auc={best_auc:.5f}")

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    return test_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["realmlp", "tabnet", "ft_transformer"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating test predictions for {args.model}...")

    if args.model == "realmlp":
        preds = generate_realmlp_test(args)
    elif args.model == "tabnet":
        preds = generate_tabnet_test(args)
    elif args.model == "ft_transformer":
        preds = generate_ft_transformer_test(args)

    test = pd.read_csv("data/test.csv")
    os.makedirs("output/submissions", exist_ok=True)
    out_path = f"output/submissions/submission_{args.model}_eng_kfold.csv"
    pd.DataFrame({"id": test["id"], "Heart Disease": preds}).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
