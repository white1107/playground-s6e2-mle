import pandas as pd
import numpy as np
import os
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch

import argparse

def train_tabnet():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain-features', action='store_true', help='Add domain knowledge features')
    args = parser.parse_args()

    print("Loading data for TabNet...")
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
    
    # Preprocessing for TabNet
    # TabNet needs categorical features as integer labels
    if args.domain_features:
        print("Adding domain-specific engineered features...")
        from src.feature_engineering import add_domain_features, get_domain_features
        X = add_domain_features(X)
        X_test = add_domain_features(X_test)
        
        # Add new numerical features to the list
        num_features.extend(get_domain_features())
    
    X = df[num_features + cat_features].copy()
    X_test = test_df[num_features + cat_features].copy()
    
    cat_idxs = []
    cat_dims = []
    for i, col in enumerate(num_features + cat_features):
        if col in cat_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            cat_idxs.append(i)
            cat_dims.append(len(le.classes_))
            
    X = X.values
    X_test_values = X_test.values
    
    # K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    test_preds = np.zeros(len(test_df))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/5")
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[val_idx], y[val_idx]
        
        clf = TabNetClassifier(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=2,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":50, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax'
        )
        
        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=['valid'],
            eval_metric=['auc'],
            max_epochs=100, patience=20,
            batch_size=1024, virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        fold_preds = clf.predict_proba(X_valid)[:, 1]
        oof_preds[val_idx] = fold_preds
        test_preds += clf.predict_proba(X_test_values)[:, 1] / 5.0
        
        print(f"Fold {fold+1} AUC: {roc_auc_score(y_valid, fold_preds):.4f}")

    print(f"Mean OOF AUC: {roc_auc_score(y, oof_preds):.4f}")
    
    # Save OOF and Sub
    os.makedirs('output/predictions', exist_ok=True)
    pd.DataFrame({'id': df['id'], 'oof_pred': oof_preds}).to_csv('output/predictions/oof_tabnet_eng.csv', index=False)
    
    os.makedirs('output/submissions', exist_ok=True)
    pd.DataFrame({'id': test_df['id'], 'Heart Disease': test_preds}).to_csv('output/submissions/submission_tabnet_eng_kfold.csv', index=False)

if __name__ == "__main__":
    train_tabnet()
