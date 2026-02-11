import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import os
import argparse
import copy
from src.feature_engineering import add_original_statistics, load_original_data, add_domain_features, get_domain_features

class RealMLP(nn.Module):
    def __init__(self, num_numerical, cat_dims, embedding_dim=4, hidden_dims=[256, 128, 64], dropout=0.2):
        super(RealMLP, self).__init__()
        
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in cat_dims
        ])
        
        cat_total_dim = len(cat_dims) * embedding_dim
        input_dim = num_numerical + cat_total_dim
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
            
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(curr_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_num, x_cat):
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        if embeddings:
            x_cat_emb = torch.cat(embeddings, dim=1)
            x = torch.cat([x_num, x_cat_emb], dim=1)
        else:
            x = x_num
            
        x = self.network(x)
        x = self.output(x)
        return self.sigmoid(x)

def train_real_mlp():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256) # 256 from Kaggle kernel
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-ens', type=int, default=8, help='Number of ensemble models per fold')
    parser.add_argument('--domain-features', action='store_true', help='Add domain knowledge features')
    parser.add_argument('--external', action='store_true')
    args = parser.parse_args()

    print("Loading data for RealMLP...")
    
    df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    original = load_original_data()
    
    # Feature names (using exact matches from train.csv)
    # Combining lists from Kaggle kernel + local definition
    num_features = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
    cat_features = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
    base_features = [c for c in num_features + cat_features if c in df.columns]

    print("Adding engineered features from Original dataset...")
    df = add_original_statistics(df, original, base_features)
    test_df = add_original_statistics(test_df, original, base_features)
    
    if args.external:
        print("Concatenating External Data...")
        # Add features to external data itself (stats on itself)
        ext_df = add_original_statistics(original.copy(), original, base_features)
        df = pd.concat([df, ext_df], axis=0).reset_index(drop=True)

    if args.domain_features:
        print("Adding domain-specific engineered features...")
        df = add_domain_features(df)
        test_df = add_domain_features(test_df)

    target = 'Heart Disease'
    
    # Identify new numerical features
    new_features = [c for c in df.columns if c.startswith('orig_')]
    if args.domain_features:
         new_features.extend(get_domain_features())
    num_features.extend(new_features)
    
    print(f"Total Numerical Features: {len(num_features)}")
    print(f"Total Categorical Features: {len(cat_features)}")
    
    # Prepare target
    # Prepare target
    # Robust handling
    print(f"Target value counts before processing:\n{df[target].value_counts(dropna=False)}")
    
    # Replace strings if present, then ensure numeric
    y_series = df[target].replace({'Presence': 1, 'Absence': 0})
    y_series = pd.to_numeric(y_series, errors='coerce')
    
    if y_series.isnull().any():
        n_nans = y_series.isnull().sum()
        print(f"WARNING: Found {n_nans} NaNs in target variable! Filling with 0.")
        y_series = y_series.fillna(0)
        
    y = y_series.values
    print(f"Target processed. Unique values: {np.unique(y)}")
    
    # Preprocessing
    # RealMLP benefit from QuantileTransformer for numerical
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    df_num = qt.fit_transform(df[num_features])
    test_num = qt.transform(test_df[num_features])
    
    # Ordinal encoding for embeddings
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_cat = oe.fit_transform(df[cat_features].astype(str)).astype(int) + 1 # +1 for NA/unknown
    test_cat = oe.transform(test_df[cat_features].astype(str)).astype(int) + 1
    
    cat_dims = [int(df_cat[:, i].max()) + 1 for i in range(df_cat.shape[1])]
    
    # K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    test_preds = np.zeros(len(test_df))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_num, y)):
        print(f"Fold {fold+1}/5")
        
        X_num_tr, X_num_va = torch.FloatTensor(df_num[train_idx]), torch.FloatTensor(df_num[val_idx])
        X_cat_tr, X_cat_va = torch.LongTensor(df_cat[train_idx]), torch.LongTensor(df_cat[val_idx])
        y_tr, y_va = torch.FloatTensor(y[train_idx]).view(-1, 1), torch.FloatTensor(y[val_idx]).view(-1, 1)
        
        train_ds = TensorDataset(X_num_tr, X_cat_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        
        
        # Ensemble Training
        ensemble_models = []
        for i_ens in range(args.n_ens):
            print(f"  Training Ensemble Model {i_ens+1}/{args.n_ens}")
            model = RealMLP(len(num_features), cat_dims, hidden_dims=[384, 384, 384, 384]).to(device) # Matches Kaggle hidden_width=384, layers=4
            optimizer = optim.Adam(model.parameters(), lr=args.lr) # Could adjust LR for deep model
            criterion = nn.BCELoss()
            
            best_loss = float('inf')
            best_model_state = None
            patience = 10
            counter = 0

            for epoch in range(args.epochs):
                model.train()
                for batch_num, batch_cat, batch_y in train_loader:
                    batch_num, batch_cat, batch_y = batch_num.to(device), batch_cat.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_num, batch_cat)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                # Simple Early Stopping logic based on validation loss
                model.eval()
                val_loss = 0
                with torch.no_grad():
                     va_num, va_cat = X_num_va.to(device), X_cat_va.to(device)
                     va_y = y_va.to(device)
                     preds = model(va_num, va_cat)
                     val_loss = criterion(preds, va_y).item()
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        # print(f"    Early stopping at epoch {epoch}")
                        break
            
            # Load best model for this ensemble member
            if best_model_state:
                model.load_state_dict(best_model_state)
            ensemble_models.append(model)
        
        # Average Predictions
        fold_val_preds = np.zeros(len(val_idx))
        fold_test_preds = np.zeros(len(test_df))
        
        for model in ensemble_models:
            model.eval()
            with torch.no_grad():
                va_num, va_cat = X_num_va.to(device), X_cat_va.to(device)
                fold_val_preds += model(va_num, va_cat).cpu().numpy().flatten() / args.n_ens
                
                ts_num, ts_cat = torch.FloatTensor(test_num).to(device), torch.LongTensor(test_cat).to(device)
                fold_test_preds += model(ts_num, ts_cat).cpu().numpy().flatten() / (args.n_ens * 5.0)

        oof_preds[val_idx] = fold_val_preds
        test_preds += fold_test_preds
        
        # Clean up to save memory
        del ensemble_models
        torch.cuda.empty_cache()
            
        print(f"Fold {fold+1} AUC: {roc_auc_score(y[val_idx], fold_val_preds):.4f}")

    print(f"Mean OOF AUC: {roc_auc_score(y, oof_preds):.4f}")
    
    # Save OOF and Sub
    os.makedirs('output/predictions', exist_ok=True)
    pd.DataFrame({'id': df['id'], 'oof_pred': oof_preds}).to_csv('output/predictions/oof_realmlp_eng.csv', index=False)
    
    os.makedirs('output/submissions', exist_ok=True)
    pd.DataFrame({'id': test_df['id'], 'Heart Disease': test_preds}).to_csv('output/submissions/submission_realmlp_eng_kfold.csv', index=False)

if __name__ == "__main__":
    train_real_mlp()
