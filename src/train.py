import argparse
import os
from datetime import datetime

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.feature_engineering import add_original_statistics, get_original_stat_features, add_domain_features, get_domain_features

# MLflow tracking (optional)
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"
if MLFLOW_ENABLED:
    try:
        from src.utils.mlflow_utils import MLflowTracker
    except ImportError:
        MLFLOW_ENABLED = False

# Define features
# Base features
NUMERICAL_FEATURES_BASE = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 'Number of vessels fluro']
CATEGORICAL_FEATURES = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Thallium']
# Engineered features
NUMERICAL_FEATURES_ENG = []
TARGET = 'Heart Disease'



def load_data(use_engineered=False, use_external=False, use_original_stats=False, validate_schema=True):
    if not os.path.exists('data/train.csv'):
        raise FileNotFoundError("data/train.csv not found")
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv') if os.path.exists('data/test.csv') else None

    # Schema validation
    if validate_schema:
        from src.schema import validate_train_data, validate_test_data
        print("Validating data schema...")
        validate_train_data(df_train)
        print("✓ Train data schema valid")
        if df_test is not None:
            validate_test_data(df_test)
            print("✓ Test data schema valid")

    if use_external:
        external_path = 'data/external/Heart_Disease_Prediction.csv'
        if os.path.exists(external_path):
            print(f"Loading external data from {external_path}...")
            df_external = pd.read_csv(external_path)
            # External data doesn't have 'id', add a dummy one to avoid issues if needed
            if 'id' not in df_external.columns:
                df_external['id'] = range(1000000, 1000000 + len(df_external))
            
            # Combine
            df_train = pd.concat([df_train, df_external], axis=0, ignore_index=True)
            print(f"✓ External data integrated. New training size: {len(df_train)}")
        else:
            print(f"Warning: External data not found at {external_path}")

    if use_engineered:
        print("Creating engineered features...")
        df_train = add_domain_features(df_train)
        if df_test is not None:
            df_test = add_domain_features(df_test)
        print("✓ Engineered features created")
        
    if use_original_stats:
        print("Adding statistics from Original dataset...")
        df_train = add_original_statistics(df_train)
        if df_test is not None:
            df_test = add_original_statistics(df_test)
        print("✓ Original statistics added")
    
    return df_train, df_test

def get_feature_names(use_engineered=False, use_original_stats=False, df=None):
    """Get feature names based on whether engineered features are used"""
    numerical = NUMERICAL_FEATURES_BASE.copy()
    categorical = CATEGORICAL_FEATURES.copy()
    
    if use_engineered:
        numerical.extend(get_domain_features())
        
    if use_original_stats and df is not None:
        stats_feats = get_original_stat_features(df)
        numerical.extend(stats_feats)
    
    return numerical, categorical

def get_pipeline(model_name, params=None, use_engineered=False, use_original_stats=False, df_sample=None, native_cats=False):
    # Get feature names
    numerical_features, categorical_features = get_feature_names(use_engineered, use_original_stats, df_sample)
    
    # Preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    if native_cats and model_name == 'cat':
        # For native CatBoost, we use a custom transformer to cast to str and fill NAs
        # to avoid sklearn's SimpleImputer type mismatch errors
        from sklearn.preprocessing import FunctionTransformer
        def cat_imputer(X):
            import pandas as pd
            # If input is already a DataFrame, use fillna, else convert and fill
            if isinstance(X, pd.DataFrame):
                return X.fillna('NA').astype(str)
            return pd.DataFrame(X).fillna('NA').astype(str)
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', FunctionTransformer(cat_imputer))
        ])
    else:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Model
    if params is None:
        params = {}
    
    if model_name == 'rf':
        default_params = {'n_estimators': 100, 'random_state': 42}
        default_params.update(params)
        model = RandomForestClassifier(**default_params)
    elif model_name == 'lgbm':
        from lightgbm import LGBMClassifier
        default_params = {'random_state': 42, 'verbose': -1, 'verbosity': -1}
        default_params.update(params)
        model = LGBMClassifier(**default_params)
    elif model_name == 'xgb':
        from xgboost import XGBClassifier
        default_params = {'random_state': 42, 'eval_metric': 'auc', 'use_label_encoder': False}
        default_params.update(params)
        model = XGBClassifier(**default_params)
    elif model_name == 'cat':
        from catboost import CatBoostClassifier
        default_params = {'random_state': 42, 'verbose': 0, 'allow_writing_files': False}
        default_params.update(params)
        model = CatBoostClassifier(**default_params)
        # CatBoost has sklearn compatibility issues with Pipeline, return separately
        return preprocessor, model
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    return pipeline, None

def optimize_hyperparameters(model_name, X_train, y_train, X_val, y_val, n_trials=50, use_engineered=False, use_original_stats=False):
    """Optimize hyperparameters using Optuna"""
    
    # We need a sample to determine feature names if original stats are used
    df_sample = pd.concat([X_train, y_train], axis=1).head() if use_original_stats else None
    
    def objective(trial):
        if model_name == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
        elif model_name == 'lgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            }
        elif model_name == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
        elif model_name == 'cat':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            }
            if use_engineered: # Add more complexity for native cats if engineered
                params['max_ctr_complexity'] = trial.suggest_int('max_ctr_complexity', 1, 4)
                params['one_hot_max_size'] = trial.suggest_int('one_hot_max_size', 2, 10)
        
        # Train model with suggested parameters
        result = get_pipeline(model_name, params, use_engineered, use_original_stats, X_train.head(), native_cats=True if model_name == 'cat' else False)
        if isinstance(result, tuple) and result[1] is not None:
            # CatBoost case
            preprocessor, model = result
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_val_transformed = preprocessor.transform(X_val)
            
            # Get categorical feature indices for the transformed data
            _, cat_feats = get_feature_names(use_engineered, use_original_stats, X_train.head())
            cat_indices = list(range(X_train_transformed.shape[1] - len(cat_feats), X_train_transformed.shape[1]))
            
            model.fit(X_train_transformed, y_train, cat_features=cat_indices)
            val_preds = model.predict_proba(X_val_transformed)[:, 1]
        else:
            # Pipeline case
            pipeline = result[0] if isinstance(result, tuple) else result
            pipeline.fit(X_train, y_train)
            val_preds = pipeline.predict_proba(X_val)[:, 1]
        
        score = roc_auc_score(y_val, val_preds)
        return score
    
    print(f"\n{'='*60}")
    print(f"Starting Optuna Hyperparameter Optimization")
    print(f"Model: {model_name.upper()}")
    print(f"Trials: {n_trials}")
    print(f"Engineered Features: {'YES' if use_engineered else 'NO'}")
    print(f"{'='*60}\n")
    
    study = optuna.create_study(direction='maximize', study_name=f'{model_name}_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n{'='*60}")
    print(f"Optimization Complete!")
    print(f"{'='*60}")
    print(f"Best AUC Score: {study.best_value:.4f}")
    print(f"Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key:20s} {value}")
    
    return study.best_params

def main():
    parser = argparse.ArgumentParser(description='Train models for Heart Disease prediction')
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'lgbm', 'xgb', 'cat'], help='Model to train: rf, lgbm, xgb, cat')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning with Optuna')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials (default: 50)')
    parser.add_argument('--engineered', action='store_true', help='Use engineered features')
    parser.add_argument('--original-stats', action='store_true', help='Use statistics from Original dataset (leak features)')
    parser.add_argument('--external', action='store_true', help='Use external UCI dataset')
    parser.add_argument('--folds', type=int, default=1, help='Number of folds for K-Fold CV (default: 1, which means single split)')
    parser.add_argument('--native-cats', action='store_true', help='Use CatBoost native categorical handling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting (default: 42)')
    parser.add_argument('--no-validate', action='store_true', help='Skip schema validation')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow tracking')
    args = parser.parse_args()

    # Initialize MLflow tracker
    tracker = None
    use_mlflow = MLFLOW_ENABLED and not args.no_mlflow
    if use_mlflow:
        tracker = MLflowTracker(experiment_name="heart-disease-prediction")

    suffix = "_eng" if args.engineered else ""
    print(f"Training Model: {args.model.upper()}{' with Engineered Features' if args.engineered else ''} ...")

    # 1. Load Data
    df_train, df_test = load_data(
        use_engineered=args.engineered,
        use_external=args.external,
        use_original_stats=args.original_stats,
        validate_schema=not args.no_validate,
    )

    # 2. Prepare Target
    # Check if target maps correctly
    if set(df_train[TARGET].unique()) == {'Presence', 'Absence'}:
        y = df_train[TARGET].map({'Presence': 1, 'Absence': 0})
    else:
        # Fallback if already 0/1 or different
        y = df_train[TARGET]
    
    # Get feature names
    numerical_features, categorical_features = get_feature_names(args.engineered, args.original_stats, df_train)
    ALL_FEATURES = numerical_features + categorical_features
    
    X = df_train[ALL_FEATURES]

    # 3. Training and Evaluation
    X_train_full = df_train[ALL_FEATURES]
    
    if args.folds > 1:
        print(f"\nPerforming {args.folds}-fold Stratified K-Fold Cross Validation...")
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        
        oof_preds = np.zeros(len(X_train_full))
        test_preds_total = np.zeros(len(df_test)) if df_test is not None else None
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y)):
            print(f"\n--- Fold {fold + 1}/{args.folds} ---")
            
            X_tr, X_va = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            
            # Tune only on first fold if tuning enabled (to save time) or skip tuning in K-Fold
            fold_params = None
            if args.tune and fold == 0:
                fold_params = optimize_hyperparameters(args.model, X_tr, y_tr, X_va, y_va, args.n_trials, args.engineered, args.original_stats)
                best_params = fold_params # Store for subsequent folds
            elif args.tune:
                fold_params = best_params
            
            result = get_pipeline(args.model, fold_params, args.engineered, args.original_stats, df_train.head(), native_cats=args.native_cats)
            
            if isinstance(result, tuple) and result[1] is not None:
                preprocessor, model = result
                X_tr_trans = preprocessor.fit_transform(X_tr)
                X_va_trans = preprocessor.transform(X_va)
                
                if args.native_cats and args.model == 'cat':
                    _, cat_feats = get_feature_names(args.engineered, args.original_stats, df_train.head())
                    cat_indices = list(range(X_tr_trans.shape[1] - len(cat_feats), X_tr_trans.shape[1]))
                    model.fit(X_tr_trans, y_tr, cat_features=cat_indices)
                else:
                    model.fit(X_tr_trans, y_tr)
                
                fold_preds = model.predict_proba(X_va_trans)[:, 1]
                if df_test is not None:
                    test_preds_total += model.predict_proba(preprocessor.transform(df_test[ALL_FEATURES]))[:, 1]
            else:
                pipeline = result[0]
                pipeline.fit(X_tr, y_tr)
                
                fold_preds = pipeline.predict_proba(X_va)[:, 1]
                if df_test is not None:
                    test_preds_total += pipeline.predict_proba(df_test[ALL_FEATURES])[:, 1]
            
            oof_preds[val_idx] = fold_preds
            fold_score = roc_auc_score(y_va, fold_preds)
            scores.append(fold_score)
            print(f"Fold {fold + 1} AUC: {fold_score:.4f}")
            
            # Save fold model? For now, we only care about OOF and averaged test preds
            
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"\n{'='*60}")
        print(f"Mean CV AUC: {mean_score:.4f} ± {std_score:.4f}")
        print(f"{'='*60}")

        # MLflow logging for K-Fold
        if use_mlflow and tracker:
            run_name = f"{args.model}-kfold-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            with tracker.start_run(run_name=run_name):
                tracker.log_params({
                    "model": args.model,
                    "engineered": args.engineered,
                    "original_stats": args.original_stats,
                    "external": args.external,
                    "folds": args.folds,
                    "n_trials": args.n_trials if args.tune else 0,
                    "seed": args.seed,
                })
                tracker.log_metrics({
                    "cv_auc_mean": mean_score,
                    "cv_auc_std": std_score,
                })
                for i, s in enumerate(scores):
                    tracker.log_metric(f"fold_{i+1}_auc", s)
                tracker.set_tag("cv_type", "kfold")
                print(f"MLflow run logged: {tracker.run_id}")
        
        # Save OOF predictions
        oof_df = pd.DataFrame({'id': df_train['id'], 'oof_pred': oof_preds})
        os.makedirs('output/predictions', exist_ok=True)
        oof_df.to_csv(f'output/predictions/oof_{args.model}{suffix}.csv', index=False)
        
        # Final Submission
        if df_test is not None:
            final_test_preds = test_preds_total / args.folds
            submission = pd.DataFrame({
                'id': df_test['id'],
                'Heart Disease': final_test_preds
            })
            os.makedirs('output/submissions', exist_ok=True)
            submission_path = f'output/submissions/submission_{args.model}{suffix}_kfold.csv'
            submission.to_csv(submission_path, index=False)
            print(f"K-Fold averaged submission saved to {submission_path}")
            
    else:
        # Original single split logic
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
        
        # 4. Hyperparameter Tuning (if enabled)
        best_params = None
        if args.tune:
            best_params = optimize_hyperparameters(args.model, X_train, y_train, X_val, y_val, args.n_trials, args.engineered, args.original_stats)
            print(f"\nTraining final model with optimized parameters...")
        else:
            print(f"\nTraining with default parameters (use --tune for optimization)...")
        
        # 5. Train Model
        result = get_pipeline(args.model, best_params, args.engineered, args.original_stats, df_train.head(), native_cats=args.native_cats)
        if isinstance(result, tuple) and result[1] is not None:
            # CatBoost case: preprocessor and model returned separately
            preprocessor, model = result
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_val_transformed = preprocessor.transform(X_val)
            
            if args.native_cats and args.model == 'cat':
                _, cat_feats = get_feature_names(args.engineered, args.original_stats, df_train.head())
                cat_indices = list(range(X_train_transformed.shape[1] - len(cat_feats), X_train_transformed.shape[1]))
                model.fit(X_train_transformed, y_train, cat_features=cat_indices)
            else:
                model.fit(X_train_transformed, y_train)
            
            # 5. Evaluate
            val_preds = model.predict_proba(X_val_transformed)[:, 1]
            score = roc_auc_score(y_val, val_preds)
            print(f"Validation AUC Score: {score:.4f}")

            # MLflow logging
            if use_mlflow and tracker:
                run_name = f"{args.model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                with tracker.start_run(run_name=run_name):
                    tracker.log_params({
                        "model": args.model,
                        "engineered": args.engineered,
                        "original_stats": args.original_stats,
                        "external": args.external,
                        "tuned": args.tune,
                        "n_trials": args.n_trials if args.tune else 0,
                        "seed": args.seed,
                        **(best_params or {}),
                    })
                    tracker.log_metrics({"val_auc": score})
                    tracker.set_tag("cv_type", "single_split")
                    print(f"MLflow run logged: {tracker.run_id}")

            # 6. Save Model
            os.makedirs('output/models', exist_ok=True)
            model_path = f"output/models/model_{args.model}{suffix}.pkl"
            joblib.dump((preprocessor, model), model_path)
            print(f"Model saved to {model_path}")

            # 7. Make Submission
            print("Generating submission...")
            if df_test is not None:
                X_test = df_test[ALL_FEATURES]
                X_test_transformed = preprocessor.transform(X_test)
                
                # Predict
                test_preds = model.predict_proba(X_test_transformed)[:, 1]
                
                # Create submission DataFrame
                submission = pd.DataFrame({
                    'id': df_test['id'],
                    'Heart Disease': test_preds
                })
                
                os.makedirs('output/submissions', exist_ok=True)
                submission_path = f'output/submissions/submission_{args.model}{suffix}.csv'
                submission.to_csv(submission_path, index=False)
                print(f"Submission saved to {submission_path}")
                
            else:
                print("Warning: data/test.csv not found. Skipping submission generation.")
        else:
            # Pipeline case (RF, LGBM, XGB)
            pipeline = result[0] if isinstance(result, tuple) else result
            pipeline.fit(X_train, y_train)
            
            # 5. Evaluate
            val_preds = pipeline.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, val_preds)
            print(f"Validation AUC Score: {score:.4f}")

            # MLflow logging
            if use_mlflow and tracker:
                run_name = f"{args.model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                with tracker.start_run(run_name=run_name):
                    tracker.log_params({
                        "model": args.model,
                        "engineered": args.engineered,
                        "external": args.external,
                        "tuned": args.tune,
                        "n_trials": args.n_trials if args.tune else 0,
                        "seed": args.seed,
                        **(best_params or {}),
                    })
                    tracker.log_metrics({"val_auc": score})
                    tracker.set_tag("cv_type", "single_split")
                    print(f"MLflow run logged: {tracker.run_id}")

            # 6. Save Model
            os.makedirs('output/models', exist_ok=True)
            model_path = f"output/models/model_{args.model}{suffix}.pkl"
            joblib.dump(pipeline, model_path)
            print(f"Model saved to {model_path}")

            # 7. Make Submission
            print("Generating submission...")
            if df_test is not None:
                X_test = df_test[ALL_FEATURES]
                
                # Predict
                test_preds = pipeline.predict_proba(X_test)[:, 1]
                
                # Create submission DataFrame
                submission = pd.DataFrame({
                    'id': df_test['id'],
                    'Heart Disease': test_preds
                })
                
                os.makedirs('output/submissions', exist_ok=True)
                submission_path = f'output/submissions/submission_{args.model}{suffix}.csv'
                submission.to_csv(submission_path, index=False)
                print(f"Submission saved to {submission_path}")
                
            else:
                print("Warning: data/test.csv not found. Skipping submission generation.")



if __name__ == "__main__":
    main()
