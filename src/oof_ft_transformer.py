"""FT-Transformer OOF generator with mini-batch training, scheduler, early stopping.

Usage:
    python -m src.oof_ft_transformer
    python -m src.oof_ft_transformer --epochs 80 --lr 1e-4 --batch-size 512
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rtdl_revisiting_models import FTTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.feature_engineering import (
    add_domain_features,
    add_original_statistics,
    get_domain_features,
    load_original_data,
)


def main():
    parser = argparse.ArgumentParser(description="FT-Transformer OOF generator")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--d-block", type=int, default=192)
    parser.add_argument("--attention-dropout", type=float, default=0.2)
    parser.add_argument("--ffn-dropout", type=float, default=0.1)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    df = pd.read_csv("data/train.csv")
    original = load_original_data()

    if df["Heart Disease"].dtype == object:
        df["Heart Disease"] = df["Heart Disease"].map({"Presence": 1, "Absence": 0})

    # Features
    num_features = ["Age", "BP", "Cholesterol", "Max HR", "ST depression", "Number of vessels fluro"]
    cat_features = ["Sex", "Chest pain type", "FBS over 120", "EKG results",
                    "Exercise angina", "Slope of ST", "Thallium"]
    base_features = [c for c in df.columns if c not in ["Heart Disease", "id"]]

    # Add original statistics
    print("Adding original statistics...")
    df = add_original_statistics(df, original, base_features)

    # Add domain features
    print("Adding domain features...")
    df = add_domain_features(df)
    domain_feats = get_domain_features()

    orig_feat_cols = [c for c in df.columns if c.startswith("orig_")]
    all_num_features = num_features + domain_feats + orig_feat_cols

    y = df["Heart Disease"].values

    # Encode categoricals
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat_raw = oe.fit_transform(df[cat_features].astype(str)).astype(int)
    cat_cardinalities = [int(X_cat_raw[:, i].max()) + 1 for i in range(X_cat_raw.shape[1])]

    X_num_raw = df[all_num_features].values.astype(np.float32)

    print(f"Numerical features: {len(all_num_features)}, Categorical features: {len(cat_features)}")
    print(f"Cat cardinalities: {cat_cardinalities}")
    print(f"Samples: {len(df)}")

    # K-Fold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_preds = np.zeros(len(df))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_num_raw, y)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{args.folds}")
        print(f"{'='*60}")

        # Per-fold scaling
        scaler = StandardScaler()
        X_num_train = scaler.fit_transform(X_num_raw[train_idx])
        X_num_val = scaler.transform(X_num_raw[val_idx])

        X_cat_train = X_cat_raw[train_idx]
        X_cat_val = X_cat_raw[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        # DataLoaders
        train_ds = TensorDataset(
            torch.tensor(X_num_train, dtype=torch.float32),
            torch.tensor(X_cat_train, dtype=torch.long),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_num_val, dtype=torch.float32),
            torch.tensor(X_cat_val, dtype=torch.long),
            torch.tensor(y_val, dtype=torch.float32),
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)

        # Model
        model = FTTransformer(
            n_cont_features=len(all_num_features),
            cat_cardinalities=cat_cardinalities,
            d_out=1,
            n_blocks=args.n_blocks,
            d_block=args.d_block,
            attention_n_heads=8,
            attention_dropout=args.attention_dropout,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=args.ffn_dropout,
            residual_dropout=0.0,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        loss_fn = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(args.epochs):
            # --- Train ---
            model.train()
            total_loss = 0.0
            n_batches = 0
            for x_num, x_cat, y_batch in train_loader:
                x_num = x_num.to(device)
                x_cat = x_cat.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)

                optimizer.zero_grad()
                logits = model(x_num, x_cat)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # --- Validate ---
            model.eval()
            val_preds_list = []
            val_labels_list = []
            with torch.no_grad():
                for x_num, x_cat, y_batch in val_loader:
                    x_num = x_num.to(device)
                    x_cat = x_cat.to(device)
                    logits = model(x_num, x_cat)
                    preds = torch.sigmoid(logits).cpu().numpy().flatten()
                    val_preds_list.append(preds)
                    val_labels_list.append(y_batch.numpy())

            val_preds_all = np.concatenate(val_preds_list)
            val_labels_all = np.concatenate(val_labels_list)
            val_auc = roc_auc_score(val_labels_all, val_preds_all)

            avg_loss = total_loss / n_batches
            lr_now = scheduler.get_last_lr()[0]

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} | loss={avg_loss:.4f} | val_auc={val_auc:.5f} | lr={lr_now:.2e}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}, best AUC={best_auc:.5f}")
                    break

        # Restore best weights
        model.load_state_dict(best_state)
        model.eval()

        # Predict validation
        val_preds_list = []
        with torch.no_grad():
            for x_num, x_cat, _ in val_loader:
                x_num = x_num.to(device)
                x_cat = x_cat.to(device)
                logits = model(x_num, x_cat)
                preds = torch.sigmoid(logits).cpu().numpy().flatten()
                val_preds_list.append(preds)

        fold_preds = np.concatenate(val_preds_list)
        oof_preds[val_idx] = fold_preds

        score = roc_auc_score(y_val, fold_preds)
        fold_scores.append(score)
        print(f"  Fold {fold + 1} Final AUC: {score:.5f} (best_val={best_auc:.5f})")

        # Free GPU memory
        del model, optimizer, scheduler, train_ds, val_ds, train_loader, val_loader
        torch.cuda.empty_cache()

    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n{'='*60}")
    print(f"FT-Transformer CV AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f"Overall OOF AUC: {overall_auc:.5f}")
    print(f"{'='*60}")

    # Save OOF
    os.makedirs("output/predictions", exist_ok=True)
    oof_path = "output/predictions/oof_ft_transformer_eng.csv"
    pd.DataFrame({"id": df["id"], "oof_pred": oof_preds}).to_csv(oof_path, index=False)
    print(f"OOF saved: {oof_path}")


if __name__ == "__main__":
    main()
