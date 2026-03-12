# Financial Health Prediction Challenge - Complete Solution

## 🎯 Problem Overview

This challenge involves predicting the Financial Health Index (FHI) for Small and Medium Enterprises (SMEs) across 4 Southern African countries (Eswatini, Lesotho, Malawi, Zimbabwe). The FHI classifies businesses into three categories: **Low, Medium, High** financial health.

**Key Challenge Details:**
- **Evaluation Metric**: F1 Score (macro for balanced class evaluation)
- **Dataset**: 9,618 training samples, 2,405 test samples
- **Features**: 39 variables including demographics, financial metrics, attitudes, and perceptions
- **Class Distribution**: Highly imbalanced (Low: 65.3%, Medium: 29.8%, High: 4.9%)

## 📊 Exploratory Data Analysis Findings

### Key Insights:
1. **Severe Class Imbalance**: High financial health businesses represent only 4.9% of the dataset
2. **Missing Data**: Significant missing values in financial access and insurance features (20-50% missing)
3. **Country Variations**: Different financial health patterns across countries
4. **Financial Ratios**: Income-to-expense and turnover ratios show strong predictive power
5. **Financial Access**: Businesses with more financial services tend to have better financial health

### Target Distribution:
- **Low**: 6,280 businesses (65.3%)
- **Medium**: 2,868 businesses (29.8%) 
- **High**: 470 businesses (4.9%)

## 🔧 Feature Engineering Strategy

### 1. Financial Ratio Features
- `income_to_expense_ratio`: Personal income vs business expenses
- `turnover_to_expense_ratio`: Business turnover vs expenses
- `income_to_turnover_ratio`: Personal income vs business turnover
- `business_efficiency`: Overall business efficiency score
- `income_per_year`: Income relative to business age

### 2. Business Age Features
- `total_business_age_months`: Complete business age in months
- `business_age_category`: Categorical age groups (startup, young, established, mature, veteran)

### 3. Financial Access Features
- `financial_access_score`: Count of financial services used
- Binary features for mobile money, credit cards, debit cards, internet banking, loan accounts

### 4. Insurance Coverage Features
- `insurance_coverage_score`: Count of insurance types
- Binary features for various insurance types

### 5. Attitude and Perception Features
- `positive_attitude_score`: Sum of positive business attitudes
- `negative_attitude_score`: Sum of concerns and negative perceptions
- `attitude_balance`: Net attitude score

### 6. Country-Specific Features
- `country_avg_income`: Country-wise median income
- `income_vs_country_avg`: Income relative to country average

## 🤖 Model Selection and Training

### Models Evaluated:
1. **XGBoost** - Gradient boosting with regularization
2. **LightGBM** - Fast gradient boosting with leaf-wise growth
3. **CatBoost** - Gradient boosting with categorical feature handling
4. **Random Forest** - Ensemble of decision trees
5. **Gradient Boosting** - Sklearn's GBM implementation
6. **Logistic Regression** - Baseline linear model
7. **SVM** - Support vector machine with RBF kernel

### Best Performing Models:
- **XGBoost**: Typically highest F1 scores with good generalization
- **LightGBM**: Fast training with competitive performance
- **CatBoost**: Excellent with categorical features

### Cross-Validation Strategy:
- **Stratified 5-fold CV** to maintain class distribution
- **F1 Macro** as primary evaluation metric
- **Random seed fixation** for reproducibility

## 📈 Performance Results

### Validation Performance (Typical):
- **XGBoost**: F1 Macro ≈ 0.45-0.55
- **LightGBM**: F1 Macro ≈ 0.43-0.53
- **CatBoost**: F1 Macro ≈ 0.42-0.52
- **Random Forest**: F1 Macro ≈ 0.40-0.50

### Ensemble Approaches:
- **Majority Voting**: Improves stability
- **Weighted Ensemble**: Based on individual model F1 scores

## 🚀 Usage Instructions

### Quick Start:
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python train_and_predict.py
```

### Advanced Options:
```bash
# Hyperparameter optimization
python hyperparameter_optimization.py

# Feature engineering only
python feature_engineering.py

# Model training only
python model_training.py
```

## 📁 File Structure

```
financial_health_prediction/
├── train_and_predict.py          # Complete pipeline script
├── feature_engineering.py        # Feature engineering class
├── model_training.py             # Model training and evaluation
├── hyperparameter_optimization.py # Hyperparameter tuning
├── requirements.txt               # Python dependencies
├── README_SOLUTION.md           # This file
├── Train.csv                    # Training dataset
├── Test.csv                     # Test dataset
├── SampleSubmission.csv         # Submission format
└── submission.csv               # Generated submission file
```

## 🎯 Submission Strategy

### To Maximize Leaderboard Performance:

1. **Start with Baseline**: Use `train_and_predict.py` for initial submission
2. **Hyperparameter Tuning**: Run `hyperparameter_optimization.py` for better models
3. **Ensemble Methods**: Combine multiple models for stability
4. **Feature Selection**: Use feature importance to remove noise
5. **Cross-Validation**: Ensure models generalize well

### Submission Limits Management:
- **10 submissions/day**: Use wisely, validate locally first
- **200 total submissions**: Save for final improvements
- **Public/Private split**: 30%/70% - focus on generalization

## 🔬 Technical Details

### Data Preprocessing:
- **Missing Values**: Median for numerical, mode for categorical
- **Categorical Encoding**: Label encoding with missing category
- **Feature Scaling**: Not required for tree-based models
- **Class Imbalance**: Class weights and balanced sampling

### Model Hyperparameters (Optimized):
```python
# XGBoost Best Parameters Example
{
    'n_estimators': 300-500,
    'max_depth': 6-8,
    'learning_rate': 0.05-0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1-0.3,
    'reg_alpha': 0.1-0.5,
    'reg_lambda': 0.1-0.5
}
```

## 📊 Feature Importance Analysis

### Top Predictive Features:
1. **Financial Ratios** (income_to_expense_ratio, turnover_to_expense_ratio)
2. **Business Age** (total_business_age_months)
3. **Financial Access Score**
4. **Country-specific features**
5. **Attitude Balance Score**
6. **Insurance Coverage Score**

## 🎖️ Improvement Strategies

### For Better Performance:
1. **Advanced Feature Engineering**
   - Interaction terms between financial ratios
   - Time-based features if temporal data available
   - Domain-specific financial health indicators

2. **Model Ensembling**
   - Stacking with meta-learner
   - Weighted voting based on validation performance
   - Blending different model families

3. **Data Augmentation**
   - Synthetic minority oversampling (SMOTE)
   - Class-balanced sampling
   - Bootstrapping techniques

4. **Cross-Validation Optimization**
   - Group K-Fold by country
   - Time-series split if temporal
   - Stratified sampling with multiple seeds

## ⚠️ Common Pitfalls to Avoid

1. **Data Leakage**: Don't use target information in features
2. **Overfitting**: Use proper cross-validation
3. **Class Imbalance**: Don't ignore minority class
4. **Reproducibility**: Set random seeds consistently
5. **Submission Format**: Match exactly the required format

## 🏆 Expected Leaderboard Performance

Based on cross-validation results:
- **Good performance**: F1 Macro > 0.50
- **Competitive performance**: F1 Macro > 0.55  
- **Top-tier performance**: F1 Macro > 0.60

## 📞 Support and Troubleshooting

### Common Issues:
1. **Memory errors**: Reduce dataset size or use incremental learning
2. **Long training times**: Use fewer iterations or simpler models
3. **Poor performance**: Check feature engineering and hyperparameters
4. **Submission format errors**: Verify column names and ID matching

### Performance Tips:
- Use LightGBM for faster training
- Enable GPU acceleration if available
- Parallelize cross-validation
- Cache feature engineering results

---

**Good luck with the competition! This solution provides a strong foundation that can be further optimized based on your specific requirements and computational resources.**
