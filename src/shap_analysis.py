import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# Define features
NUMERICAL_FEATURES = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 'Number of vessels fluro']
CATEGORICAL_FEATURES = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Thallium']
TARGET = 'Heart Disease'

def create_engineered_features(df):
    """Create domain-knowledge based features"""
    df = df.copy()
    df['RPP'] = df['BP'] * df['Max HR']
    df['Electrical_Stress'] = df['ST depression'] * df['Slope of ST']
    df['Metabolic_Score'] = (df['BP'] > 130).astype(int) + \
                            (df['Cholesterol'] > 200).astype(int) + \
                            (df['FBS over 120'] == 1).astype(int)
    df['MaxHR_Rel_Age'] = df['Max HR'] / (220 - df['Age'])
    return df

def main():
    parser = argparse.ArgumentParser(description='SHAP analysis for Heart Disease models')
    parser.add_argument('--model', type=str, default='cat', choices=['rf', 'lgbm', 'xgb', 'cat'], help='Model to analyze')
    parser.add_argument('--engineered', action='store_true', help='Use engineered features')
    args = parser.parse_args()

    suffix = "_eng" if args.engineered else ""
    
    # Load data
    print("Loading data...")
    df_train = pd.read_csv('data/train.csv')
    
    if args.engineered:
        print("Creating engineered features...")
        df_train = create_engineered_features(df_train)
        engineered_features = ['RPP', 'Electrical_Stress', 'Metabolic_Score', 'MaxHR_Rel_Age']
        numerical_features = NUMERICAL_FEATURES + engineered_features
    else:
        numerical_features = NUMERICAL_FEATURES
    
    ALL_FEATURES = numerical_features + CATEGORICAL_FEATURES

    # Target
    y = df_train['Heart Disease'].map({'Presence': 1, 'Absence': 0})

    # Prepare features (same preprocessing as train.py)
    X_num = df_train[numerical_features].copy()
    X_cat = df_train[CATEGORICAL_FEATURES].copy()

    # Impute
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_num_imputed = num_imputer.fit_transform(X_num)
    X_cat_imputed = cat_imputer.fit_transform(X_cat)

    # Encode categorical
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_cat_encoded = encoder.fit_transform(X_cat_imputed)

    # Combine
    X = np.hstack([X_num_imputed, X_cat_encoded])
    X_df = pd.DataFrame(X, columns=ALL_FEATURES)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)

    # Load model
    model_path = f'output/models/model_{args.model}{suffix}.pkl'
    print(f"\nLoading model from {model_path}...")
    try:
        preprocessor, model = joblib.load(model_path)
        print(f"✓ {args.model.upper()} model loaded")
        model_name = args.model.upper()
    except:
        print(f"Error: Could not load model from {model_path}")
        return

    # Create SHAP explainer
    print(f"\nCreating SHAP explainer for {model_name}...")
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values (use subset for speed)
    print("Calculating SHAP values...")
    sample_size = min(500, len(X_val))
    X_sample = X_val.sample(n=sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, get positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class

    print(f"✓ SHAP values calculated for {sample_size} samples")

    # 1. Summary Plot (Feature Importance)
    print("\n[1/3] Creating Summary Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f'SHAP Summary Plot - {model_name}{suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'output/shap_summary{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: output/shap_summary{suffix}.png")
    plt.close()

    # 2. Decision Plot (Individual predictions)
    print("\n[2/3] Creating Decision Plot...")
    sample_indices = [0, 1, 2, 3, 4]  # First 5 samples
    plt.figure(figsize=(10, 8))
    shap.decision_plot(
        explainer.expected_value,
        shap_values[sample_indices],
        X_sample.iloc[sample_indices],
        show=False,
        feature_display_range=slice(None, -20, -1)  # Show top features
    )
    plt.title(f'SHAP Decision Plot - {model_name}{suffix}\n(First 5 validation samples)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'output/shap_decision{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: output/shap_decision{suffix}.png")
    plt.close()

    # 3. Feature Importance (Bar plot)
    print("\n[3/3] Creating Feature Importance Plot...")
    shap_importance = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': ALL_FEATURES,
        'Importance': shap_importance
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 10))
    plt.barh(range(len(feature_importance_df)), feature_importance_df['Importance'])
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
    plt.xlabel('Mean |SHAP value|', fontsize=12)
    plt.title(f'Feature Importance - {model_name}{suffix}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'output/shap_importance{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: output/shap_importance{suffix}.png")
    plt.close()

    print("\n" + "="*60)
    print("SHAP Analysis Complete!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  1. output/shap_summary{suffix}.png    - Summary plot (feature importance + impact)")
    print(f"  2. output/shap_decision{suffix}.png   - Decision plot (individual predictions)")
    print(f"  3. output/shap_importance{suffix}.png - Feature importance bar chart")
    print(f"\nTop 10 Most Important Features:")
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"  {row['Feature']:25s} {row['Importance']:.4f}")

if __name__ == "__main__":
    main()
