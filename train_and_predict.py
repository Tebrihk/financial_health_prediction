#!/usr/bin/env python3
"""
Financial Health Prediction - Complete Training and Prediction Script

This script provides a complete pipeline for:
1. Data loading and preprocessing
2. Feature engineering
3. Model training with cross-validation
4. Hyperparameter optimization
5. Prediction generation
6. Submission file creation

Author: ML Engineer
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer

class FinancialHealthPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer()
        self.trainer = ModelTrainer(random_state=random_state)
        self.final_model = None
        self.feature_columns = None
        
    def load_data(self):
        """Load training and test datasets"""
        print("Loading datasets...")
        train_df = pd.read_csv('Train.csv')
        test_df = pd.read_csv('Test.csv')
        sample_submission = pd.read_csv('SampleSubmission.csv')
        
        print(f"Training data: {train_df.shape}")
        print(f"Test data: {test_df.shape}")
        print(f"Sample submission: {sample_submission.shape}")
        
        return train_df, test_df, sample_submission
    
    def analyze_data(self, train_df):
        """Basic data analysis"""
        print("\n" + "="*60)
        print("DATA ANALYSIS")
        print("="*60)
        
        # Target distribution
        print("\nTarget distribution:")
        target_dist = train_df['Target'].value_counts()
        for target, count in target_dist.items():
            print(f"  {target}: {count} ({count/len(train_df)*100:.1f}%)")
        
        # Missing values
        print(f"\nMissing values summary:")
        missing = train_df.isnull().sum()
        missing_cols = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing_cols) > 0:
            print(f"  Columns with missing values: {len(missing_cols)}")
            print(f"  Top 5 missing:")
            for col, count in missing_cols.head().items():
                pct = count/len(train_df)*100
                print(f"    {col}: {count} ({pct:.1f}%)")
        else:
            print("  No missing values found!")
        
        # Country distribution
        print(f"\nCountry distribution:")
        country_dist = train_df['country'].value_counts()
        for country, count in country_dist.items():
            print(f"  {country}: {count} ({count/len(train_df)*100:.1f}%)")
    
    def preprocess_data(self, train_df, test_df):
        """Apply feature engineering to both datasets"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        # Apply feature engineering to training data
        train_processed = self.feature_engineer.fit_transform(train_df)
        
        # Apply same transformations to test data
        test_processed = self.feature_engineer.transform(test_df)
        
        # Store feature columns
        self.feature_columns = self.feature_engineer.feature_columns
        
        print(f"Training data shape after engineering: {train_processed.shape}")
        print(f"Test data shape after engineering: {test_processed.shape}")
        print(f"Total features created: {len(self.feature_columns)}")
        
        return train_processed, test_processed
    
    def train_models(self, train_processed):
        """Train and evaluate multiple models"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Prepare data
        X = train_processed.drop(['ID', 'Target'], axis=1)
        y = train_processed['Target']
        
        # Train and evaluate models
        results = self.trainer.train_and_evaluate_models(X, y)
        
        # Cross-validation
        cv_results = self.trainer.cross_validate_models(X, y, cv_folds=5)
        
        print(f"\nBest model validation F1: {self.trainer.best_score:.4f}")
        
        # Train final model on all data
        self.final_model = self.trainer.train_final_model(X, y)
        
        return results, cv_results
    
    def generate_predictions(self, test_processed, sample_submission):
        """Generate predictions for test set"""
        print("\n" + "="*60)
        print("GENERATING PREDICTIONS")
        print("="*60)
        
        # Get test features
        X_test = test_processed[self.feature_columns]
        
        # Make predictions
        predictions = self.final_model.predict(X_test)
        
        # Create submission file
        submission = pd.DataFrame({
            'ID': test_processed['ID'],
            'Target': predictions
        })
        
        # Verify format matches sample submission
        assert submission.shape == sample_submission.shape, "Submission shape mismatch!"
        assert list(submission.columns) == list(sample_submission.columns), "Column mismatch!"
        
        print(f"Generated {len(predictions)} predictions")
        print(f"Prediction distribution:")
        pred_dist = pd.Series(predictions).value_counts()
        for target, count in pred_dist.items():
            print(f"  {target}: {count} ({count/len(predictions)*100:.1f}%)")
        
        return submission
    
    def save_submission(self, submission, filename='submission.csv'):
        """Save submission file"""
        submission.to_csv(filename, index=False)
        print(f"\nSubmission saved as: {filename}")
        
        # Display first few rows
        print(f"\nFirst 5 predictions:")
        print(submission.head())
    
    def generate_model_report(self, train_processed):
        """Generate comprehensive model report"""
        print("\n" + "="*60)
        print("MODEL REPORT")
        print("="*60)
        
        # Feature importance
        X = train_processed.drop(['ID', 'Target'], axis=1)
        feature_importance = self.trainer.feature_importance_analysis(X, X.columns.tolist())
        
        if feature_importance is not None:
            # Save feature importance
            feature_importance.to_csv('feature_importance.csv', index=False)
            print("\nFeature importance saved as: feature_importance.csv")
        
        # Model performance summary
        print(f"\nFinal model: {type(self.final_model).__name__}")
        print(f"Best validation F1 score: {self.trainer.best_score:.4f}")
        
        return feature_importance
    
    def run_complete_pipeline(self):
        """Run the complete training and prediction pipeline"""
        print("Starting Financial Health Prediction Pipeline")
        print("="*60)
        
        # Load data
        train_df, test_df, sample_submission = self.load_data()
        
        # Analyze data
        self.analyze_data(train_df)
        
        # Preprocess data
        train_processed, test_processed = self.preprocess_data(train_df, test_df)
        
        # Train models
        results, cv_results = self.train_models(train_processed)
        
        # Generate predictions
        submission = self.generate_predictions(test_processed, sample_submission)
        
        # Save submission
        self.save_submission(submission)
        
        # Generate report
        self.generate_model_report(train_processed)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Files generated:")
        print("  - submission.csv (main submission file)")
        print("  - feature_importance.csv (feature analysis)")
        
        return submission, results, cv_results

def main():
    """Main execution function"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize predictor
    predictor = FinancialHealthPredictor(random_state=42)
    
    # Run complete pipeline
    submission, results, cv_results = predictor.run_complete_pipeline()
    
    return predictor, submission, results, cv_results

if __name__ == "__main__":
    predictor, submission, results, cv_results = main()
