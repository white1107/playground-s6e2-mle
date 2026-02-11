# %% [markdown]
# # Playground Series Season 6 Episode 2: Heart Disease Prediction - EDA
# 
# This script is designed to be run as an interactive Python script (using VS Code's Jupyter extension or Jupytext).
# Cells are marked with `# %%`.
#
# **Goal:** Understand the data distribution, check for missing values, and identify potential features.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)

# %% [markdown]
# ## 1. Load Data
# We'll load the train and test datasets.

# %%
import os

# Robust data loading
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

train_path = os.path.join(DATA_DIR, 'train.csv')
test_path = os.path.join(DATA_DIR, 'test.csv')

if not os.path.exists(train_path):
    # Fallback if running interactively where __file__ might be different
    train_path = '../data/train.csv'
    test_path = '../data/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")

# %% [markdown]
# ## 2. Basic Inspection
# Let's look at the first few rows, data types, and missing values.

# %%
print("--- Train Head ---")
print(train_df.head())

print("\n--- Train Info ---")
train_df.info()

print("\n--- Missing Values (Train) ---")
missing_train = train_df.isnull().sum()
missing_train = missing_train[missing_train > 0]
if len(missing_train) > 0:
    print(missing_train)
else:
    print("No missing values found in train set!")

# %% [markdown]
# ## 3. Target Distribution
# Is the target balanced or imbalanced?

# %%
target_col = 'Heart Disease'

plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x=target_col)
plt.title('Target Variable Distribution (Heart Disease)')
plt.show()

print(train_df[target_col].value_counts(normalize=True))

# %% [markdown]
# ## 4. Numerical Features Distribution
# Let's look at the distribution of numerical features.

# %%
features = [col for col in train_df.columns if col not in ['id', target_col]]
num_features = train_df[features].select_dtypes(include=[np.number]).columns.tolist()

print(f"Numerical Features: {len(num_features)}")
print(num_features)

# %%
# Histograms for numerical features
n_cols = 3
n_rows = (len(num_features) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.histplot(data=train_df, x=col, hue=target_col, kde=True, element="step", ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')

# Hide empty subplots
for i in range(len(num_features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Correlation Matrix
# Check for linear relationships between features and target.

# %%
# Encode target for correlation if needed
if train_df[target_col].dtype == 'object':
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_df[target_col + '_encoded'] = le.fit_transform(train_df[target_col])
    corr_target = target_col + '_encoded'
else:
    corr_target = target_col

# Select numeric columns explicitly for correlation
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
if corr_target not in numeric_cols:
    numeric_cols.append(corr_target)

corr_matrix = train_df[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# ## 6. Categorical Features
# Check unique values for categorical columns (including numeric ones that represent categories).

# %%
# Numeric columns that are actually categorical
categorical_features = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
                        'Exercise angina', 'Slope of ST', 'Thallium']

print(f"Categorical-like Features: {categorical_features}\n")

for col in categorical_features:
    print(f"\n--- {col} ---")
    print(train_df[col].value_counts().sort_index())
    
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_df, x=col, hue=target_col)
    plt.title(f'{col} vs {target_col}')
    plt.xticks(rotation=0)
    plt.show()

