import pandas as pd
import numpy as np
import optuna
import os
import argparse
from sklearn.metrics import roc_auc_score

def load_predictions(model_names, suffix="_eng", type="oof"):
    folder = "output/predictions" if type == "oof" else "output/submissions"
    prefix = "oof_" if type == "oof" else "submission_"
    
    preds = {}
    for model in model_names:
        # Standard filenames from train.py:
        # OOF: oof_model_eng.csv
        # SUB: submission_model_eng_kfold.csv
        
        # Try with _kfold if it's a submission, or without if it's OOF
        if type == "sub":
            filename = f"{prefix}{model}{suffix}_kfold.csv"
            # Fallback to no _kfold
            if not os.path.exists(os.path.join(folder, filename)):
                filename = f"{prefix}{model}{suffix}.csv"
        else:
            filename = f"{prefix}{model}{suffix}.csv"
            # Fallback to _kfold
            if not os.path.exists(os.path.join(folder, filename)):
                filename = f"{prefix}{model}{suffix}_kfold.csv"
             
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            col_name = "oof_pred" if type == "oof" else "Heart Disease"
            preds[model] = df[col_name].values
            print(f"Loaded {type} for {model}: {filename}")
        else:
            print(f"Warning: {type} for {model} not found at {filepath}")
    
    return preds

def objective(trial, model_names, oof_preds, y_true):
    weights = []
    for model in model_names:
        weights.append(trial.suggest_float(f"w_{model}", 0, 1))
    
    # Normalize weights
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    
    # Calculate ensemble prediction
    ensemble_pred = np.zeros(len(y_true))
    for i, model in enumerate(model_names):
        ensemble_pred += weights[i] * oof_preds[model]
    
    return roc_auc_score(y_true, ensemble_pred)

def main():
    parser = argparse.ArgumentParser(description='Weighted Ensemble Blending')
    parser.add_argument('--models', type=str, default='cat,xgb,lgbm', help='Comma separated model names')
    parser.add_argument('--suffix', type=str, default='_eng', help='Suffix for files')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials')
    args = parser.parse_args()

    model_names = args.models.split(',')
    
    print("Loading data for target...")
    train_df = pd.read_csv('data/train.csv')
    
    # Load external data if it exists to match OOF shape
    external_path = 'data/external/Heart_Disease_Prediction.csv'
    if os.path.exists(external_path):
        ext_df = pd.read_csv(external_path)
        train_df = pd.concat([train_df, ext_df], axis=0).reset_index(drop=True)
        print(f"✓ Combined external data. Total target size: {len(train_df)}")

    # Prepare target
    if set(train_df['Heart Disease'].unique()) == {'Presence', 'Absence'}:
        y_true = train_df['Heart Disease'].map({'Presence': 1, 'Absence': 0}).values
    else:
        y_true = train_df['Heart Disease'].values
    
    print("\nLoading OOF predictions...")
    oof_preds = load_predictions(model_names, args.suffix, type="oof")
    
    if len(oof_preds) < 2:
        print("Error: Need at least 2 models for ensembling")
        return

    # Filter model names to those actually loaded
    active_models = list(oof_preds.keys())
    
    print("\n============================================================")
    print("Finding Optimal Weights with Optuna")
    print(f"Models: {active_models}")
    print("============================================================\n")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, active_models, oof_preds, y_true), n_trials=args.n_trials)
    
    print("\n============================================================")
    print("Optimization Result")
    print(f"Best Blended AUC: {study.best_value:.4f}")
    
    best_weights = study.best_params
    total_w = sum(best_weights.values())
    normalized_weights = {k: v/total_w for k, v in best_weights.items()}
    
    for model, weight in normalized_weights.items():
        print(f"  {model}: {weight:.4f}")
    print("============================================================\n")
    
    print("Loading test submissions...")
    sub_preds = load_predictions(active_models, args.suffix, type="sub")
    
    if len(sub_preds) == len(active_models):
        print("\nGenerating Ensembled Submission...")
        ensemble_sub = np.zeros(len(next(iter(sub_preds.values()))))
        for model in active_models:
            ensemble_sub += normalized_weights[f"w_{model}"] * sub_preds[model]
            
        test_df = pd.read_csv('data/test.csv')
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Heart Disease': ensemble_sub
        })
        
        output_path = 'output/submissions/submission_ensemble.csv'
        submission.to_csv(output_path, index=False)
        print(f"✓ Ensemble submission saved to {output_path}")
    else:
        print("\nWarning: Some test submissions were missing. Skipping final file generation.")

if __name__ == "__main__":
    main()
