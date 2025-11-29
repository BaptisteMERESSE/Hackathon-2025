"""
Data Exploration Script for PISA Math Score Prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
print("Loading data...")
X_train = pd.read_csv('../data/X_train.csv', index_col=0)
y_train = pd.read_csv('../data/y_train.csv', index_col=0)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"\nFirst few rows of y_train:\n{y_train.head()}")

# Basic statistics
print(f"\n{'='*80}")
print("TARGET VARIABLE STATISTICS")
print(f"{'='*80}")
print(y_train.describe())
print(f"\nNumber of zeros in target: {(y_train['MathScore'] == 0).sum()}")
print(f"Percentage of zeros: {(y_train['MathScore'] == 0).sum() / len(y_train) * 100:.2f}%")

# Feature types
print(f"\n{'='*80}")
print("FEATURE INFORMATION")
print(f"{'='*80}")
print(f"Total features: {X_train.shape[1]}")
print(f"\nData types:\n{X_train.dtypes.value_counts()}")

# Missing values
print(f"\n{'='*80}")
print("MISSING VALUES")
print(f"{'='*80}")
missing_counts = X_train.isnull().sum()
missing_pct = (missing_counts / len(X_train)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_counts,
    'Missing_Percentage': missing_pct
}).sort_values('Missing_Percentage', ascending=False)

print(f"Features with missing values: {(missing_counts > 0).sum()}")
print(f"\nTop 20 features with most missing values:")
print(missing_df.head(20))

# Numeric vs categorical features
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"\n{'='*80}")
print("FEATURE TYPES")
print(f"{'='*80}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

if categorical_features:
    print(f"\nCategorical features:")
    for feat in categorical_features[:10]:
        print(f"  - {feat}: {X_train[feat].nunique()} unique values")

# Save feature lists
np.save('../data/numeric_features.npy', numeric_features)
np.save('../data/categorical_features.npy', categorical_features)

# Correlation analysis (sample for speed)
print(f"\n{'='*80}")
print("CORRELATION ANALYSIS (sample of 10000 rows)")
print(f"{'='*80}")
sample_size = min(10000, len(X_train))
X_sample = X_train.iloc[:sample_size][numeric_features]
y_sample = y_train.iloc[:sample_size]

# Merge for correlation
merged = pd.concat([X_sample, y_sample], axis=1)
correlations = merged.corr()['MathScore'].sort_values(ascending=False)

print(f"\nTop 20 features most correlated with MathScore:")
print(correlations.head(20))

print(f"\nBottom 20 features least correlated with MathScore:")
print(correlations.tail(20))

# Save exploration results
with open('../data/exploration_summary.txt', 'w') as f:
    f.write(f"Dataset Shape: {X_train.shape}\n")
    f.write(f"Target Variable Range: [{y_train['MathScore'].min()}, {y_train['MathScore'].max()}]\n")
    f.write(f"Target Variable Mean: {y_train['MathScore'].mean()}\n")
    f.write(f"Numeric Features: {len(numeric_features)}\n")
    f.write(f"Categorical Features: {len(categorical_features)}\n")
    f.write(f"Features with Missing Values: {(missing_counts > 0).sum()}\n")

print(f"\n{'='*80}")
print("Exploration complete! Results saved.")
print(f"{'='*80}")

