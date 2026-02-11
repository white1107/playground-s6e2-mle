import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# Load data
df_train = pd.read_csv('data/train.csv')

# Target
y = df_train['Heart Disease'].map({'Presence': 1, 'Absence': 0})

# Define features
NUMERICAL_FEATURES = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 'Number of vessels fluro']
CATEGORICAL_FEATURES = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Thallium']

print("=" * 60)
print("スコア比較実験")
print("=" * 60)

# Experiment 1: Baseline (数値のみ、fillna(0))
print("\n[実験1] ベースライン: 数値6個のみ + fillna(0)")
X1 = df_train[NUMERICAL_FEATURES].fillna(0)
X_train, X_val, y_train, y_val = train_test_split(X1, y, test_size=0.2, random_state=42, stratify=y)
rf1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf1.fit(X_train, y_train)
score1 = roc_auc_score(y_val, rf1.predict_proba(X_val)[:, 1])
print(f"スコア: {score1:.4f}")

# Experiment 2: 数値のみ、median imputation
print("\n[実験2] 数値6個のみ + median imputation")
X2 = df_train[NUMERICAL_FEATURES].copy()
imputer = SimpleImputer(strategy='median')
X2_imputed = imputer.fit_transform(X2)
X_train, X_val, y_train, y_val = train_test_split(X2_imputed, y, test_size=0.2, random_state=42, stratify=y)
rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf2.fit(X_train, y_train)
score2 = roc_auc_score(y_val, rf2.predict_proba(X_val)[:, 1])
print(f"スコア: {score2:.4f}")
print(f"改善: +{score2-score1:.4f}")

# Experiment 3: 全特徴量(数値+カテゴリ)、fillna(0)
print("\n[実験3] 全特徴量13個 + fillna(0)")
X3 = df_train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].fillna(0)
X_train, X_val, y_train, y_val = train_test_split(X3, y, test_size=0.2, random_state=42, stratify=y)
rf3 = RandomForestClassifier(n_estimators=100, random_state=42)
rf3.fit(X_train, y_train)
score3 = roc_auc_score(y_val, rf3.predict_proba(X_val)[:, 1])
print(f"スコア: {score3:.4f}")
print(f"改善: +{score3-score1:.4f}")

# Experiment 4: 全特徴量 + 適切な前処理
print("\n[実験4] 全特徴量13個 + 適切な前処理")
X4_num = df_train[NUMERICAL_FEATURES].copy()
X4_cat = df_train[CATEGORICAL_FEATURES].copy()

# Impute
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
X4_num_imputed = num_imputer.fit_transform(X4_num)
X4_cat_imputed = cat_imputer.fit_transform(X4_cat)

# Encode categorical
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X4_cat_encoded = encoder.fit_transform(X4_cat_imputed)

# Combine
X4 = np.hstack([X4_num_imputed, X4_cat_encoded])
X_train, X_val, y_train, y_val = train_test_split(X4, y, test_size=0.2, random_state=42, stratify=y)
rf4 = RandomForestClassifier(n_estimators=100, random_state=42)
rf4.fit(X_train, y_train)
score4 = roc_auc_score(y_val, rf4.predict_proba(X_val)[:, 1])
print(f"スコア: {score4:.4f}")
print(f"改善: +{score4-score1:.4f}")

print("\n" + "=" * 60)
print("結論")
print("=" * 60)
print(f"ベースライン → 最終: {score1:.4f} → {score4:.4f} (+{score4-score1:.4f})")
print(f"\n最大の改善要因:")
print(f"  - カテゴリ特徴量の追加: +{score3-score1:.4f} ポイント")
print(f"  - 適切な前処理: +{score4-score3:.4f} ポイント")
