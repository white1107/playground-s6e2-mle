import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from rtdl_revisiting_models import FTTransformer

import argparse

def train_ft_transformer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain-features', action='store_true', help='Add domain knowledge features')
    args = parser.parse_args()

    print("Loading data for FT-Transformer...")
    df = pd.read_csv('data/train.csv')
    ext_path = 'data/external/Heart_Disease_Prediction.csv'
    if os.path.exists(ext_path):
        ext_df = pd.read_csv(ext_path)
        df = pd.concat([df, ext_df], axis=0).reset_index(drop=True)
    
    test_df = pd.read_csv('data/test.csv')
    
    # Feature names
    num_features = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
    cat_features = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST']
    target = 'Heart Disease'
    
    # Prepare target
    y = df[target].map({'Presence': 1, 'Absence': 0}).values
    
    # Preprocessing
    if args.domain_features:
        print("Adding domain-specific engineered features...")
        from src.feature_engineering import add_domain_features, get_domain_features
        df = add_domain_features(df)
        test_df = add_domain_features(test_df)
        
        # Add new numerical features to the list
        num_features.extend(get_domain_features())

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_features])
    X_num_test = scaler.transform(test_df[num_features])
    
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_cat = oe.fit_transform(df[cat_features].astype(str)).astype(int)
    X_cat_test = oe.transform(test_df[cat_features].astype(str)).astype(int)
    
    cat_cardinalities = [len(np.unique(X_cat[:, i])) for i in range(X_cat.shape[1])]
    
    # K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    test_preds = np.zeros(len(test_df))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_num, y)):
        print(f"Fold {fold+1}/5")
        
        model = FTTransformer(
            n_cont_features=len(num_features),
            cat_cardinalities=cat_cardinalities,
            d_out=1,
            **FTTransformer.get_default_kwargs()
        ).to(device)
        
        # Training logic here...
        # For brevity in 100 trials, we use a simple training loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss()
        
        x_num_tr = torch.tensor(X_num[train_idx], dtype=torch.float32).to(device)
        x_cat_tr = torch.tensor(X_cat[train_idx], dtype=torch.long).to(device)
        y_tr = torch.tensor(y[train_idx], dtype=torch.float32).to(device).unsqueeze(1)
        
        x_num_va = torch.tensor(X_num[val_idx], dtype=torch.float32).to(device)
        x_cat_va = torch.tensor(X_cat[val_idx], dtype=torch.long).to(device)
        
        # Simple training
        model.train()
        for epoch in range(20): # Fewer epochs for quick demonstration
            optimizer.zero_grad()
            logits = model(x_num_tr, x_cat_tr)
            loss = loss_fn(logits, y_tr)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            logits_va = model(x_num_va, x_cat_va)
            fold_preds = torch.sigmoid(logits_va).cpu().numpy().flatten()
            oof_preds[val_idx] = fold_preds
            
            x_num_ts = torch.tensor(X_num_test, dtype=torch.float32).to(device)
            x_cat_ts = torch.tensor(X_cat_test, dtype=torch.long).to(device)
            logits_ts = model(x_num_ts, x_cat_ts)
            test_preds += torch.sigmoid(logits_ts).cpu().numpy().flatten() / 5.0
            
        print(f"Fold {fold+1} AUC: {roc_auc_score(y[val_idx], fold_preds):.4f}")

    print(f"Mean OOF AUC: {roc_auc_score(y, oof_preds):.4f}")
    
    # Save OOF and Sub
    os.makedirs('output/predictions', exist_ok=True)
    pd.DataFrame({'id': df['id'], 'oof_pred': oof_preds}).to_csv('output/predictions/oof_ft_transformer_eng.csv', index=False)
    
    os.makedirs('output/submissions', exist_ok=True)
    pd.DataFrame({'id': test_df['id'], 'Heart Disease': test_preds}).to_csv('output/submissions/submission_ft_transformer_eng_kfold.csv', index=False)

if __name__ == "__main__":
    train_ft_transformer()
