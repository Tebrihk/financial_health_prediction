import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import FeatureEngineer

class HyperparameterOptimizer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer()
        self.best_models = {}
        self.best_scores = {}
        
    def prepare_data(self):
        """Load and prepare data for optimization"""
        print("Loading and preparing data...")
        
        # Load data
        train_df = pd.read_csv('Train.csv')
        
        # Feature engineering
        train_processed = self.feature_engineer.fit_transform(train_df)
        
        # Prepare features and target
        X = train_processed.drop(['ID', 'Target'], axis=1)
        y = train_processed['Target']
        
        print(f"Data prepared: {X.shape}")
        return X, y
    
    def create_scorer(self):
        """Create F1 macro scorer"""
        return make_scorer(f1_score, average='macro')
    
    def optimize_xgboost(self, X, y, n_iter=50):
        """Optimize XGBoost hyperparameters"""
        print("\nOptimizing XGBoost...")
        
        # Define parameter space
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1),
            'min_child_weight': randint(1, 10)
        }
        
        # Create model
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Randomized search
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=self.create_scorer(),
            cv=cv,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        search.fit(X, y)
        
        self.best_models['XGBoost'] = search.best_estimator_
        self.best_scores['XGBoost'] = search.best_score_
        
        print(f"Best XGBoost F1: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")
        
        return search.best_estimator_, search.best_score_, search.best_params_
    
    def optimize_lightgbm(self, X, y, n_iter=50):
        """Optimize LightGBM hyperparameters"""
        print("\nOptimizing LightGBM...")
        
        # Define parameter space
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1),
            'min_child_samples': randint(5, 100),
            'num_leaves': randint(20, 100)
        }
        
        # Create model
        lgb_model = lgb.LGBMClassifier(
            objective='multiclass',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        # Randomized search
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            lgb_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=self.create_scorer(),
            cv=cv,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        search.fit(X, y)
        
        self.best_models['LightGBM'] = search.best_estimator_
        self.best_scores['LightGBM'] = search.best_score_
        
        print(f"Best LightGBM F1: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")
        
        return search.best_estimator_, search.best_score_, search.best_params_
    
    def optimize_catboost(self, X, y, n_iter=30):
        """Optimize CatBoost hyperparameters"""
        print("\nOptimizing CatBoost...")
        
        # Define parameter space (smaller due to CatBoost being slower)
        param_dist = {
            'iterations': randint(100, 500),
            'depth': randint(4, 10),
            'learning_rate': uniform(0.01, 0.3),
            'l2_leaf_reg': uniform(1, 10),
            'border_count': randint(32, 255)
        }
        
        # Create model
        cat_model = cb.CatBoostClassifier(
            loss_function='MultiClass',
            random_seed=self.random_state,
            verbose=False
        )
        
        # Randomized search
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            cat_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=self.create_scorer(),
            cv=cv,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        search.fit(X, y)
        
        self.best_models['CatBoost'] = search.best_estimator_
        self.best_scores['CatBoost'] = search.best_score_
        
        print(f"Best CatBoost F1: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")
        
        return search.best_estimator_, search.best_score_, search.best_params_
    
    def optimize_random_forest(self, X, y, n_iter=30):
        """Optimize Random Forest hyperparameters"""
        print("\nOptimizing Random Forest...")
        
        # Define parameter space
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Create model
        rf_model = RandomForestClassifier(
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Randomized search
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            rf_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=self.create_scorer(),
            cv=cv,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        search.fit(X, y)
        
        self.best_models['RandomForest'] = search.best_estimator_
        self.best_scores['RandomForest'] = search.best_score_
        
        print(f"Best Random Forest F1: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")
        
        return search.best_estimator_, search.best_score_, search.best_params_
    
    def optimize_all_models(self, X, y):
        """Optimize all models"""
        print("Starting hyperparameter optimization for all models...")
        print("="*60)
        
        results = {}
        
        # Optimize each model
        try:
            results['XGBoost'] = self.optimize_xgboost(X, y, n_iter=50)
        except Exception as e:
            print(f"XGBoost optimization failed: {e}")
        
        try:
            results['LightGBM'] = self.optimize_lightgbm(X, y, n_iter=50)
        except Exception as e:
            print(f"LightGBM optimization failed: {e}")
        
        try:
            results['CatBoost'] = self.optimize_catboost(X, y, n_iter=30)
        except Exception as e:
            print(f"CatBoost optimization failed: {e}")
        
        try:
            results['RandomForest'] = self.optimize_random_forest(X, y, n_iter=30)
        except Exception as e:
            print(f"Random Forest optimization failed: {e}")
        
        # Summary
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        for model_name, (model, score, params) in results.items():
            print(f"{model_name}: F1 = {score:.4f}")
        
        # Find best overall model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k][1])
            best_model, best_score, best_params = results[best_model_name]
            
            print(f"\nBest overall model: {best_model_name}")
            print(f"Best F1 score: {best_score:.4f}")
            print(f"Best parameters: {best_params}")
            
            return best_model, best_model_name, best_score, results
        
        return None, None, None, results
    
    def save_best_models(self, filename='best_models_results.csv'):
        """Save optimization results"""
        if not self.best_scores:
            print("No results to save!")
            return
        
        results_df = pd.DataFrame([
            {'Model': model, 'Best_F1_Score': score}
            for model, score in self.best_scores.items()
        ]).sort_values('Best_F1_Score', ascending=False)
        
        results_df.to_csv(filename, index=False)
        print(f"Optimization results saved to: {filename}")
        print(results_df)

def main():
    """Main optimization function"""
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(random_state=42)
    
    # Prepare data
    X, y = optimizer.prepare_data()
    
    # Optimize all models
    best_model, best_model_name, best_score, results = optimizer.optimize_all_models(X, y)
    
    # Save results
    optimizer.save_best_models()
    
    return optimizer, best_model, best_model_name, best_score, results

if __name__ == "__main__":
    optimizer, best_model, best_model_name, best_score, results = main()
