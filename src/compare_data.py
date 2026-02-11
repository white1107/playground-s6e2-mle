import pandas as pd
import numpy as np
import os

def compare_datasets():
    # Load data
    train_df = pd.read_csv('data/train.csv')
    ext_df = pd.read_csv('data/external/Heart_Disease_Prediction.csv')
    
    # Drop id from train
    train_df = train_df.drop(columns=['id'])
    
    print(f"Dataset Sizes:")
    print(f"  Competition Train: {len(train_df)}")
    print(f"  External UCI:     {len(ext_df)}")
    
    # Target distribution
    train_target = train_df['Heart Disease'].value_counts(normalize=True).to_dict()
    ext_target = ext_df['Heart Disease'].value_counts(normalize=True).to_dict()
    
    print(f"\nTarget Distribution ('Heart Disease'):")
    print(f"  Train:    {train_target}")
    print(f"  External: {ext_target}")
    
    # Numerical features comparison
    num_features = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
    
    print(f"\nNumerical Feature Stats (Mean / Std):")
    stats = []
    for feat in num_features:
        t_mean, t_std = train_df[feat].mean(), train_df[feat].std()
        e_mean, e_std = ext_df[feat].mean(), ext_df[feat].std()
        diff_mean = ((e_mean - t_mean) / t_mean) * 100
        stats.append({
            'Feature': feat,
            'Train Mean': f"{t_mean:.2f}",
            'Ext Mean': f"{e_mean:.2f}",
            'Diff %': f"{diff_mean:+.2f}%",
            'Train Std': f"{t_std:.2f}",
            'Ext Std': f"{e_std:.2f}"
        })
    
    print(pd.DataFrame(stats).to_string(index=False))
    
    # Categorical features comparison
    cat_features = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
    
    print(f"\nCategorical Feature Values (Top 1 frequency):")
    cat_stats = []
    for feat in cat_features:
        t_top = train_df[feat].value_counts(normalize=True).iloc[0]
        e_top = ext_df[feat].value_counts(normalize=True).iloc[0]
        cat_stats.append({
            'Feature': feat,
            'Train Top%': f"{t_top*100:.2f}%",
            'Ext Top%': f"{e_top*100:.2f}%"
        })
    print(pd.DataFrame(cat_stats).to_string(index=False))

if __name__ == "__main__":
    compare_datasets()
