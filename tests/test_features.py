
import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import add_original_statistics, load_original_data

def test_original_stats():
    # 1. Create dummy original data
    original_df = pd.DataFrame({
        'Age': [50, 50, 50, 60, 60],
        'Heart Disease': [1, 1, 0, 0, 0] # Presence=1, Absence=0
    })
    
    # 2. Create dummy train data
    train_df = pd.DataFrame({
        'Age': [50, 60, 70],
        'id': [1, 2, 3]
    })
    
    # Expected stats for Age=50: mean=2/3=0.666, count=3
    # Expected stats for Age=60: mean=0/2=0.0, count=2
    # Expected stats for Age=70: mean=Global Mean (2/5=0.4)
    
    # 3. Apply feature engineering
    base_features = ['Age']
    result_df = add_original_statistics(train_df, original_df, base_features)
    
    print("\n--- Original Data ---")
    print(original_df)
    print("\n--- Processed Train Data ---")
    print(result_df)
    
    # 4. Assertions
    # Check 'orig_Age_mean'
    assert 'orig_Age_mean' in result_df.columns
    
    # Age 50
    age50_row = result_df[result_df['Age'] == 50].iloc[0]
    assert np.isclose(age50_row['orig_Age_mean'], 2/3)
    assert age50_row['orig_Age_count'] == 3
    
    # Age 60
    age60_row = result_df[result_df['Age'] == 60].iloc[0]
    assert np.isclose(age60_row['orig_Age_mean'], 0.0)
    assert age60_row['orig_Age_count'] == 2
    
    # Age 70 (Not in original) -> Should be filled with global mean
    age70_row = result_df[result_df['Age'] == 70].iloc[0]
    global_mean = original_df['Heart Disease'].mean()
    assert np.isclose(age70_row['orig_Age_mean'], global_mean)
    assert age70_row['orig_Age_count'] == 0

if __name__ == "__main__":
    test_original_stats()
