import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    original = pd.read_csv('data/external/Heart_Disease_Prediction.csv')
    
    # Preprocess Original
    # Ensure column names match if they are different (they seem to be consistent based on previous context)
    # Map target if necessary, but we are looking at features mainly
    
    return train, test, original

def compare_numerical(train, test, original, features):
    print("\n--- Numerical Distribution Check (KS Test) ---")
    print(f"{'Feature':<20} | {'Train vs Test (p-val)':<25} | {'Train vs Orig (p-val)':<25}")
    print("-" * 75)
    
    results = []
    for col in features:
        if col not in original.columns:
            continue
            
        # Drop NaNs for KS test
        tr_vals = train[col].dropna()
        te_vals = test[col].dropna()
        or_vals = original[col].dropna()
        
        # KS Test
        # Null hypothesis: two samples are drawn from the same distribution
        # If p-value is small (< 0.05), we reject the null hypothesis -> distributions are different
        
        ks_tr_te = ks_2samp(tr_vals, te_vals)
        ks_tr_or = ks_2samp(tr_vals, or_vals)
        
        print(f"{col:<20} | {ks_tr_te.pvalue:.4f} ({'Diff' if ks_tr_te.pvalue < 0.05 else 'Same'})     | {ks_tr_or.pvalue:.4f} ({'Diff' if ks_tr_or.pvalue < 0.05 else 'Same'})")
        results.append({
            'Feature': col,
            'Train_vs_Test_p': ks_tr_te.pvalue,
            'Train_vs_Orig_p': ks_tr_or.pvalue
        })
    return pd.DataFrame(results)

def compare_categorical(train, test, original, features):
    print("\n--- Categorical Distribution Check (Top 3 Value Counts) ---")
    
    for col in features:
        if col not in original.columns:
            continue
            
        print(f"\nFeature: {col}")
        
        # Calculate value counts as percentages
        tr_vc = train[col].value_counts(normalize=True).sort_index()
        te_vc = test[col].value_counts(normalize=True).sort_index()
        or_vc = original[col].value_counts(normalize=True).sort_index()
        
        # Combine into a dataframe for easier viewing
        comp_df = pd.DataFrame({
            'Train': tr_vc,
            'Test': te_vc,
            'Original': or_vc
        }).fillna(0)
        
        print(comp_df.sort_values('Train', ascending=False).head(3))

def main():
    try:
        train, test, original = load_data()
    except FileNotFoundError:
        print("Error: Data files not found. Make sure 'data/train.csv', 'data/test.csv', and 'data/external/Heart_Disease_Prediction.csv' exist.")
        return

    # Identify features
    cat_features = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Thallium', 'Number of vessels fluro'] # Adding Thallium based on EDA context
    num_features = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
    
    # Ensure Thallium, Number of vessels fluro are treated appropriately
    # The user's provided code treated them as categorical implicitly or explicitly
    
    compare_numerical(train, test, original, num_features)
    compare_categorical(train, test, original, cat_features)

if __name__ == "__main__":
    main()
