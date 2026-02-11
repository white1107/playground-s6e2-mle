import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_train = pd.read_csv('data/train.csv')
df_train['Heart Disease Binary'] = df_train['Heart Disease'].map({'Presence': 1, 'Absence': 0})

# All features
NUMERICAL_FEATURES = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 'Number of vessels fluro']
CATEGORICAL_FEATURES = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Thallium']
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Calculate correlation matrix
df_corr = df_train[ALL_FEATURES].copy()
correlation_matrix = df_corr.corr()

print("="*70)
print("Thallium の他の特徴量との相関")
print("="*70)

# Get Thallium correlations
thallium_corr = correlation_matrix['Thallium'].drop('Thallium').sort_values(ascending=False)

print("\n強い相関がある特徴量 (|相関| > 0.3):")
for feature, corr in thallium_corr.items():
    if abs(corr) > 0.3:
        direction = "↑ Positive" if corr > 0 else "↓ Negative"
        print(f"  {feature:25s} {corr:7.4f}  {direction}")

# Visualize Thallium correlations
plt.figure(figsize=(10, 8))
colors = ['red' if x < 0 else 'green' for x in thallium_corr.values]
plt.barh(range(len(thallium_corr)), thallium_corr.values, color=colors, alpha=0.7)
plt.yticks(range(len(thallium_corr)), thallium_corr.index)
plt.xlabel('Correlation with Thallium', fontsize=12)
plt.title('Thallium の他特徴量との相関\n(多重共線性チェック)', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('output/thallium_correlation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: output/thallium_correlation.png")
plt.close()

# Check unique values
print("\n" + "="*70)
print("Thallium の値の分布")
print("="*70)
print(f"\nユニーク値: {sorted(df_train['Thallium'].unique())}")
print(f"値の数: {df_train['Thallium'].nunique()}")
print("\n値ごとのカウント:")
print(df_train['Thallium'].value_counts().sort_index())

print("\n値ごとの心臓病発症率:")
disease_by_thallium = df_train.groupby('Thallium')['Heart Disease Binary'].agg(['mean', 'count'])
disease_by_thallium.columns = ['Disease Rate', 'Count']
disease_by_thallium['Disease Rate'] = disease_by_thallium['Disease Rate'] * 100
print(disease_by_thallium)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution
df_train['Thallium'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.7)
axes[0].set_title('Thallium の値の分布', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Thallium Value')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.3)

# Disease rate by Thallium
disease_by_thallium['Disease Rate'].plot(kind='bar', ax=axes[1], color='coral', alpha=0.7)
axes[1].set_title('Thallium値ごとの心臓病発症率', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Thallium Value')
axes[1].set_ylabel('Disease Rate (%)')
axes[1].axhline(y=df_train['Heart Disease Binary'].mean()*100, color='red', linestyle='--', label='Overall Average')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/thallium_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: output/thallium_analysis.png")
plt.close()

print("\n" + "="*70)
print("結論: なぜThalliumはSHAPで低いのか?")
print("="*70)
print("""
考えられる理由:

1. **多重共線性**: Thalliumの情報が他の特徴量(特にChest pain type,
   Exercise angina, Number of vessels fluro)と重複している可能性

2. **非線形性**: 相関は線形関係のみ測定。Thalliumの効果は他の特徴量と
   組み合わさって初めて強くなる可能性(交互作用)

3. **モデルの選択**: CatBoostは他の特徴量を優先的に使用し、Thalliumは
   補助的な役割になっている可能性

4. **情報の冗長性**: Max HRやChest pain typeで既に十分な情報が得られて
   いるため、Thalliumの追加的な貢献度が低い

→ 相関が高い ≠ モデルにとって重要、という好例!
""")
