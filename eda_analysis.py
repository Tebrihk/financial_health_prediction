import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')

print("=== DATASET OVERVIEW ===")
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Number of features: {train_df.shape[1] - 2}")  # Excluding ID and Target

print("\n=== TARGET DISTRIBUTION ===")
target_dist = train_df['Target'].value_counts()
print(target_dist)
print(f"Class distribution percentages:")
for cls, count in target_dist.items():
    print(f"  {cls}: {count/len(train_df)*100:.1f}%")

print("\n=== MISSING VALUES ANALYSIS ===")
missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()

print("Training set missing values:")
for col, missing in missing_train.items():
    if missing > 0:
        print(f"  {col}: {missing} ({missing/len(train_df)*100:.1f}%)")

print("\nTest set missing values:")
for col, missing in missing_test.items():
    if missing > 0:
        print(f"  {col}: {missing} ({missing/len(test_df)*100:.1f}%)")

print("\n=== DATA TYPES ANALYSIS ===")
print("Training set data types:")
dtypes = train_df.dtypes.value_counts()
for dtype, count in dtypes.items():
    print(f"  {dtype}: {count} columns")

# Identify categorical and numerical features
categorical_features = []
numerical_features = []

for col in train_df.columns:
    if col not in ['ID', 'Target']:
        if train_df[col].dtype == 'object':
            categorical_features.append(col)
        else:
            numerical_features.append(col)

print(f"\nCategorical features: {len(categorical_features)}")
print(f"Numerical features: {len(numerical_features)}")

print("\n=== CATEGORICAL FEATURES ANALYSIS ===")
for col in categorical_features[:5]:  # Show first 5
    unique_vals = train_df[col].nunique()
    print(f"\n{col}: {unique_vals} unique values")
    if unique_vals <= 10:
        print(train_df[col].value_counts())

print("\n=== NUMERICAL FEATURES SUMMARY ===")
print(train_df[numerical_features].describe())

print("\n=== COUNTRY DISTRIBUTION ===")
country_dist = train_df['country'].value_counts()
print(country_dist)

print("\n=== TARGET BY COUNTRY ===")
country_target = pd.crosstab(train_df['country'], train_df['Target'])
print(country_target)
print("\nTarget percentages by country:")
for country in country_dist.index:
    country_data = train_df[train_df['country'] == country]
    print(f"\n{country}:")
    for target in ['Low', 'Medium', 'High']:
        if target in country_data['Target'].values:
            pct = (country_data['Target'] == target).sum() / len(country_data) * 100
            print(f"  {target}: {pct:.1f}%")

# Correlation analysis for numerical features
print("\n=== CORRELATION ANALYSIS ===")
numerical_data = train_df[numerical_features + ['Target']].copy()
le = LabelEncoder()
numerical_data['Target_encoded'] = le.fit_transform(numerical_data['Target'])

correlation_matrix = numerical_data[numerical_features + ['Target_encoded']].corr()
print("Top correlations with Target:")
target_corr = correlation_matrix['Target_encoded'].abs().sort_values(ascending=False)
print(target_corr.head(10))

# Feature importance preview
print("\n=== QUICK FEATURE IMPORTANCE ===")
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Prepare data for quick model
X = train_df.drop(['ID', 'Target'], axis=1)
y = train_df['Target']

# Handle categorical variables
X_encoded = X.copy()
for col in categorical_features:
    if col in X.columns:
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].fillna('missing'))

# Handle missing numerical values
X_encoded = X_encoded.fillna(X_encoded.median())

# Quick random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_encoded, y)

feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 important features:")
print(feature_importance.head(15))

print("\n=== BUSINESS INSIGHTS ===")
# Financial ratios
train_copy = train_df.copy()
train_copy['income_to_expense_ratio'] = train_copy['personal_income'] / (train_copy['business_expenses'] + 1)
train_copy['turnover_to_expense_ratio'] = train_copy['business_turnover'] / (train_copy['business_expenses'] + 1)

print("Financial ratios by target:")
for target in ['Low', 'Medium', 'High']:
    subset = train_copy[train_copy['Target'] == target]
    print(f"\n{target}:")
    print(f"  Avg income/expense ratio: {subset['income_to_expense_ratio'].median():.2f}")
    print(f"  Avg turnover/expense ratio: {subset['turnover_to_expense_ratio'].median():.2f}")
    print(f"  Avg business age: {subset['business_age_years'].median():.1f} years")

print("\n=== EDA COMPLETE ===")
