import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print("Loading data...")
df_train = pd.read_csv('data/train.csv')

# Convert target to binary
df_train['Heart Disease Binary'] = df_train['Heart Disease'].map({'Presence': 1, 'Absence': 0})

# All features
NUMERICAL_FEATURES = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 'Number of vessels fluro']
CATEGORICAL_FEATURES = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Thallium']
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Select features + target
df_corr = df_train[ALL_FEATURES + ['Heart Disease Binary']].copy()

# Calculate correlation matrix
print("\nCalculating correlation matrix...")
correlation_matrix = df_corr.corr()

# Extract correlation with target
target_correlation = correlation_matrix['Heart Disease Binary'].drop('Heart Disease Binary').sort_values(ascending=False)

print("\n" + "="*60)
print("Correlation with Heart Disease")
print("="*60)
for feature, corr in target_correlation.items():
    direction = "↑ Positive" if corr > 0 else "↓ Negative"
    print(f"{feature:25s} {corr:7.4f}  {direction}")

# 1. Correlation Heatmap (Full)
print("\n[1/3] Creating Full Correlation Heatmap...")
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - All Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('output/correlation_heatmap_full.png', dpi=300, bbox_inches='tight')
print("✓ Saved: output/correlation_heatmap_full.png")
plt.close()

# 2. Correlation with Target (Bar plot)
print("\n[2/3] Creating Target Correlation Bar Plot...")
plt.figure(figsize=(10, 8))
colors = ['red' if x < 0 else 'green' for x in target_correlation.values]
plt.barh(range(len(target_correlation)), target_correlation.values, color=colors, alpha=0.7)
plt.yticks(range(len(target_correlation)), target_correlation.index)
plt.xlabel('Correlation with Heart Disease', fontsize=12)
plt.title('Feature Correlation with Heart Disease\n(Green=Positive, Red=Negative)', 
          fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('output/correlation_target.png', dpi=300, bbox_inches='tight')
print("✓ Saved: output/correlation_target.png")
plt.close()

# 3. Top Correlations (Absolute value)
print("\n[3/3] Creating Top Correlations Plot...")
top_n = 10
top_correlations = target_correlation.abs().sort_values(ascending=False).head(top_n)
top_features = top_correlations.index.tolist()

# Get actual correlation values (with sign)
top_corr_values = target_correlation[top_features]

plt.figure(figsize=(10, 6))
colors = ['red' if x < 0 else 'green' for x in top_corr_values.values]
bars = plt.barh(range(len(top_corr_values)), top_corr_values.values, color=colors, alpha=0.7)
plt.yticks(range(len(top_corr_values)), top_features)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.title(f'Top {top_n} Features by Correlation with Heart Disease', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.gca().invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_corr_values.values)):
    plt.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}', 
             va='center', ha='left' if val > 0 else 'right', fontsize=10)

plt.tight_layout()
plt.savefig('output/correlation_top.png', dpi=300, bbox_inches='tight')
print("✓ Saved: output/correlation_top.png")
plt.close()

# 4. Numerical Features Only (Scatter Matrix style)
print("\n[4/4] Creating Numerical Features Correlation...")
df_numerical = df_train[NUMERICAL_FEATURES + ['Heart Disease Binary']].copy()
corr_numerical = df_numerical.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_numerical, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numerical Features Only', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('output/correlation_numerical.png', dpi=300, bbox_inches='tight')
print("✓ Saved: output/correlation_numerical.png")
plt.close()

print("\n" + "="*60)
print("Correlation Analysis Complete!")
print("="*60)
print("\nGenerated files:")
print("  1. output/correlation_heatmap_full.png - Full correlation matrix")
print("  2. output/correlation_target.png       - Correlation with target (all features)")
print("  3. output/correlation_top.png          - Top 10 features by correlation")
print("  4. output/correlation_numerical.png    - Numerical features only")

print("\n" + "="*60)
print("Key Insights:")
print("="*60)
print("\nStrongest Positive Correlations (↑ Higher value → Higher disease risk):")
for feature, corr in target_correlation.head(3).items():
    if corr > 0:
        print(f"  • {feature:25s} {corr:+.4f}")

print("\nStrongest Negative Correlations (↓ Higher value → Lower disease risk):")
for feature, corr in target_correlation.tail(3).items():
    if corr < 0:
        print(f"  • {feature:25s} {corr:+.4f}")
