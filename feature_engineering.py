import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def create_financial_ratios(self, df):
        """Create financial ratio features"""
        df = df.copy()
        
        # Income to expense ratio
        df['income_to_expense_ratio'] = df['personal_income'] / (df['business_expenses'] + 1)
        
        # Turnover to expense ratio  
        df['turnover_to_expense_ratio'] = df['business_turnover'] / (df['business_expenses'] + 1)
        
        # Income to turnover ratio
        df['income_to_turnover_ratio'] = df['personal_income'] / (df['business_turnover'] + 1)
        
        # Business efficiency score
        df['business_efficiency'] = (df['business_turnover'] + 1) / (df['business_expenses'] + 1)
        
        # Personal income relative to business age
        df['income_per_year'] = df['personal_income'] / (df['business_age_years'] + 1)
        
        return df
    
    def create_business_age_features(self, df):
        """Create business age related features"""
        df = df.copy()
        
        # Total business age in months
        df['total_business_age_months'] = df['business_age_years'] * 12 + df['business_age_months'].fillna(0)
        
        # Business age categories
        df['business_age_category'] = pd.cut(df['total_business_age_months'], 
                                           bins=[0, 12, 36, 60, 120, np.inf],
                                           labels=['startup', 'young', 'established', 'mature', 'veteran'])
        
        return df
    
    def create_financial_access_features(self, df):
        """Create financial access and inclusion features"""
        df = df.copy()
        
        # Count of financial services used
        financial_services = ['has_mobile_money', 'has_credit_card', 'has_debit_card', 
                           'has_internet_banking', 'has_loan_account']
        
        # Convert to binary (1 if has service, 0 if doesn't)
        for col in financial_services:
            if col in df.columns:
                df[col + '_binary'] = df[col].apply(lambda x: 1 if str(x).lower() in ['have now', 'yes'] else 0)
        
        # Financial access score
        binary_cols = [col + '_binary' for col in financial_services if col + '_binary' in df.columns]
        df['financial_access_score'] = df[binary_cols].sum(axis=1)
        
        return df
    
    def create_insurance_features(self, df):
        """Create insurance related features"""
        df = df.copy()
        
        # Count of insurance types
        insurance_types = ['has_insurance', 'motor_vehicle_insurance', 'medical_insurance', 'funeral_insurance']
        
        for col in insurance_types:
            if col in df.columns:
                df[col + '_binary'] = df[col].apply(lambda x: 1 if str(x).lower() in ['have now', 'yes'] else 0)
        
        binary_insurance = [col + '_binary' for col in insurance_types if col + '_binary' in df.columns]
        df['insurance_coverage_score'] = df[binary_insurance].sum(axis=1)
        
        return df
    
    def create_attitude_features(self, df):
        """Create attitude and perception features"""
        df = df.copy()
        
        # Positive attitudes
        positive_attitudes = ['attitude_stable_business_environment', 'attitude_satisfied_with_achievement',
                            'attitude_more_successful_next_year', 'perception_insurance_important']
        
        # Negative attitudes/concerns
        negative_attitudes = ['attitude_worried_shutdown', 'perception_insurance_doesnt_cover_losses',
                           'perception_cannot_afford_insurance', 'perception_insurance_companies_dont_insure_businesses_like_yours',
                           'future_risk_theft_stock', 'current_problem_cash_flow']
        
        # Create positive attitude score
        df['positive_attitude_score'] = 0
        for col in positive_attitudes:
            if col in df.columns:
                df['positive_attitude_score'] += df[col].apply(lambda x: 1 if str(x).lower() in ['yes', "yes, always"] else 0)
        
        # Create negative attitude score
        df['negative_attitude_score'] = 0
        for col in negative_attitudes:
            if col in df.columns:
                df['negative_attitude_score'] += df[col].apply(lambda x: 1 if str(x).lower() in ['yes'] else 0)
        
        # Attitude balance
        df['attitude_balance'] = df['positive_attitude_score'] - df['negative_attitude_score']
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        df = df.copy()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove ID and Target from encoding
        if 'ID' in categorical_features:
            categorical_features.remove('ID')
        if 'Target' in categorical_features:
            categorical_features.remove('Target')
        
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            # Fill missing values with 'missing'
            df[col] = df[col].fillna('missing')
            
            if fit:
                # Fit and transform
                try:
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                except:
                    # Handle unseen categories
                    df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                # Transform only
                try:
                    df[col] = self.label_encoders[col].transform(df[col])
                except:
                    # Handle unseen categories
                    df[col] = 0  # Default to 0 for unseen categories
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        df = df.copy()
        
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'ID' and col != 'Target':
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns (already encoded), fill with mode
        categorical_cols = [col for col in df.columns if df[col].dtype == 'int64' and col not in ['ID', 'Target']]
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)
        
        return df
    
    def create_country_features(self, df):
        """Create country-specific features"""
        df = df.copy()
        
        # Country encoding will be handled by the categorical encoder
        # But we can add country-specific statistics
        
        # Country-wise median income (for reference)
        if 'personal_income' in df.columns and 'country' in df.columns:
            country_income_map = df.groupby('country')['personal_income'].median().to_dict()
            df['country_avg_income'] = df['country'].map(country_income_map)
            
            # Income relative to country average
            df['income_vs_country_avg'] = df['personal_income'] / (df['country_avg_income'] + 1)
        
        return df
    
    def fit_transform(self, df):
        """Full feature engineering pipeline for training"""
        print("Starting feature engineering...")
        
        # Create features
        df = self.create_financial_ratios(df)
        df = self.create_business_age_features(df)
        df = self.create_financial_access_features(df)
        df = self.create_insurance_features(df)
        df = self.create_attitude_features(df)
        df = self.create_country_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Store feature columns (excluding ID and Target)
        self.feature_columns = [col for col in df.columns if col not in ['ID', 'Target']]
        
        print(f"Feature engineering complete. Created {len(self.feature_columns)} features.")
        
        return df
    
    def transform(self, df):
        """Feature engineering for test data"""
        print("Transforming test data...")
        
        # Create features
        df = self.create_financial_ratios(df)
        df = self.create_business_age_features(df)
        df = self.create_financial_access_features(df)
        df = self.create_insurance_features(df)
        df = self.create_attitude_features(df)
        df = self.create_country_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=False)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Ensure same columns as training
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Keep only the feature columns
        df = df[self.feature_columns]
        
        print(f"Test data transformation complete.")
        
        return df

# Test the feature engineering
if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('Train.csv')
    test_df = pd.read_csv('Test.csv')
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Transform training data
    train_processed = fe.fit_transform(train_df)
    
    # Transform test data
    test_processed = fe.transform(test_df)
    
    print(f"Processed training data shape: {train_processed.shape}")
    print(f"Processed test data shape: {test_processed.shape}")
    
    # Show some of the new features
    new_features = [col for col in fe.feature_columns if any(keyword in col.lower() 
                   for keyword in ['ratio', 'score', 'binary', 'category', 'vs'])]
    print(f"\nCreated {len(new_features)} new engineered features:")
    for feature in new_features[:10]:  # Show first 10
        print(f"  - {feature}")
