import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def run_stacking():
    print("Layer 1: Loading OOF predictions...")
    
    # Models to stack: 3 GBDTs + 3 DL models
    models = ['cat', 'xgb', 'lgbm', 'realmlp', 'tabnet', 'ft_transformer']
    
    # Target loading
    train_df = pd.read_csv('data/train.csv')
    external_path = 'data/external/Heart_Disease_Prediction.csv'
    if os.path.exists(external_path):
        ext_df = pd.read_csv(external_path)
        train_df = pd.concat([train_df, ext_df], axis=0).reset_index(drop=True)
    
    if set(train_df['Heart Disease'].unique()) == {'Presence', 'Absence'}:
        y_true = train_df['Heart Disease'].map({'Presence': 1, 'Absence': 0}).values
    else:
        y_true = train_df['Heart Disease'].values
        
    oof_data = {}
    for m in models:
        # Try to load OOF predictions
        path = f'output/predictions/oof_{m}_eng.csv'
        
        if os.path.exists(path):
            oof_data[m] = pd.read_csv(path)['oof_pred'].values
            print(f"  ✓ Loaded OOF for {m} from {path}")
        else:
            print(f"  ✗ Warning: OOF for {m} not found at {path}")

    if len(oof_data) < 2:
        print("Error: Not enough OOF files for stacking.")
        return

    # Create Meta-features
    X_meta = np.column_stack([oof_data[m] for m in models if m in oof_data])
    print(f"\nMeta-features shape: {X_meta.shape}")
    
    print("\nLayer 2: Training Meta-Model (Logistic Regression)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    meta_scores = []
    meta_oof_preds = np.zeros(len(y_true))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y_true)):
        X_tr, X_va = X_meta[train_idx], X_meta[val_idx]
        y_tr, y_va = y_true[train_idx], y_true[val_idx]
        
        meta_model = LogisticRegression(max_iter=1000)
        meta_model.fit(X_tr, y_tr)
        
        meta_oof_preds[val_idx] = meta_model.predict_proba(X_va)[:, 1]
        meta_scores.append(roc_auc_score(y_va, meta_oof_preds[val_idx]))
        print(f"Fold {fold+1} Meta-model AUC: {meta_scores[-1]:.4f}")
    
    print(f"\n🎯 Stacked CV AUC: {np.mean(meta_scores):.4f} ± {np.std(meta_scores):.4f}")
    
    # Final Meta-Model on all data
    final_meta_model = LogisticRegression(max_iter=1000)
    final_meta_model.fit(X_meta, y_true)
    
    print("\nGenerating Final Submission...")
    sub_data = {}
    for m in models:
        path = f'output/submissions/submission_{m}_eng_kfold.csv'
        
        if os.path.exists(path):
            sub_data[m] = pd.read_csv(path)['Heart Disease'].values
            print(f"  ✓ Loaded Submission for {m} from {path}")
        else:
            print(f"  ✗ Warning: Submission for {m} not found at {path}")
            
    if len(sub_data) < 2:
        print("Error: Not enough submission files for stacking.")
        return

    X_sub = np.column_stack([sub_data[m] for m in models if m in sub_data])
    final_preds = final_meta_model.predict_proba(X_sub)[:, 1]
    
    test_df = pd.read_csv('data/test.csv')
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Heart Disease': final_preds
    })
    
    output_path = 'output/submissions/submission_stacked_all.csv'
    submission.to_csv(output_path, index=False)
    print(f"✓ Stacked submission saved to {output_path}")

if __name__ == "__main__":
    run_stacking()
