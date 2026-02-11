import pandas as pd
import numpy as np

def load_original_data():
    """Load and preprocess original dataset"""
    original = pd.read_csv('data/external/Heart_Disease_Prediction.csv')
    # Standardize target
    if 'Heart Disease' in original.columns:
        original['Heart Disease'] = original['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    return original

def add_original_statistics(df, original=None, base_features=None):
    """
    Add statistical features (mean, median, std, skew, count) from the Original dataset.
    This mimics the 'leak' features used in high-scoring Kaggle kernels.
    """
    if original is None:
        original = load_original_data()
        
    df_temp = df.copy()
    
    # If base_features not provided, try to infer common features
    if base_features is None:
        # Features that exist in both datasets
        base_features = [c for c in df.columns if c in original.columns and c != 'Heart Disease' and c != 'id']
    
    # Calculate stats from original data
    stats_dict = {}
    for col in base_features:
        if col in original.columns:
            # We map target to 0/1 before aggregation if not already done
            if original['Heart Disease'].dtype == object:
                 original['Heart Disease'] = original['Heart Disease'].map({'Presence': 1, 'Absence': 0})

            stats = original.groupby(col)['Heart Disease'].agg(['mean', 'median', 'std', 'skew', 'count']).reset_index()
            stats.columns = [col] + [f"orig_{col}_{s}" for s in ['mean', 'median', 'std', 'skew', 'count']]
            stats_dict[col] = stats
    
    new_cols = []
    for col in base_features:
        if col in stats_dict:
            df_temp = df_temp.merge(stats_dict[col], on=col, how='left')
            
            # Keep track of new columns for fillna
            stats_cols = [f"orig_{col}_{s}" for s in ['mean', 'median', 'std', 'skew', 'count']]
            new_cols.extend(stats_cols)
            
            fill_values = {
                f"orig_{col}_mean": original['Heart Disease'].mean(),
                f"orig_{col}_median": original['Heart Disease'].median(),
                f"orig_{col}_std": 0,
                f"orig_{col}_skew": 0,
                f"orig_{col}_count": 0
            }
            df_temp = df_temp.fillna(value=fill_values)
            
    return df_temp

def add_domain_features(df):
    """
    Create domain-knowledge based features (Medical & Statistical interactions)
    """
    df = df.copy()
    
    # 1. Hemodynamic Features
    # Rate Pressure Product (RPP): Index of myocardial oxygen consumption
    df['Rate_Pressure_Product'] = df['BP'] * df['Max HR']
    
    # 2. Risk Scores
    # Metabolic Score (Simple additive risk count)
    # BP > 130, Cholesterol > 200, FBS > 0 (over 120)
    df['Metabolic_Score'] = (
        (df['BP'] > 130).astype(int) + 
        (df['Cholesterol'] > 200).astype(int) + 
        (df['FBS over 120'] > 0).astype(int)
    )
    
    # 3. Heart Rate Analysis
    # Max HR relative to Age-predicted max (220 - Age)
    # Avoid division by zero if Age is weird, though Age > 0 usually
    df['MaxHR_Rel_Age'] = df['Max HR'] / (220 - df['Age'])
    
    # 4. Stress Interaction
    # Electrical instability combined with slope
    df['Electrical_Stress'] = df['ST depression'] * df['Slope of ST']
    
    # 5. Advanced Interactions
    df['MaxHR_x_Age'] = df['Max HR'] * df['Age']
    df['BP_x_Cholesterol'] = df['BP'] * df['Cholesterol']
    
    # 6. Binning
    df['Age_Bin'] = (df['Age'] // 10).astype(int)
    
    return df

def get_original_stat_features(df):
    """Return list of features starting with 'orig_'"""
    return [c for c in df.columns if c.startswith('orig_')]

def get_domain_features():
    """Return list of domain engineered features"""
    return [
        'Rate_Pressure_Product', 'Metabolic_Score', 'MaxHR_Rel_Age', 
        'Electrical_Stress', 'MaxHR_x_Age', 'BP_x_Cholesterol', 'Age_Bin'
    ]
