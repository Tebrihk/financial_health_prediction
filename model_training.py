import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import FeatureEngineer

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        
    def define_models(self):
        """Define all models to test"""
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multiclass',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            
            'CatBoost': cb.CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                loss_function='MultiClass',
                class_weights=[1, 1, 1],  # Will be adjusted
                random_seed=self.random_state,
                verbose=False
            ),
            
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            ),
            
            'LogisticRegression': LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            ),
            
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=self.random_state
            )
        }
        
        return models
    
    def train_and_evaluate_models(self, X_train, y_train, X_val, y_val):
        """Train and evaluate multiple models"""
        models = self.define_models()
        results = {}
        
        print("Training and evaluating models...")
        print("=" * 60)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_val)
                
                # Calculate F1 score (macro for balanced evaluation across classes)
                f1_macro = f1_score(y_val, y_pred, average='macro')
                f1_weighted = f1_score(y_val, y_pred, average='weighted')
                
                results[name] = {
                    'model': model,
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted,
                    'predictions': y_pred
                }
                
                print(f"  F1 Macro: {f1_macro:.4f}")
                print(f"  F1 Weighted: {f1_weighted:.4f}")
                
                # Update best model
                if f1_macro > self.best_score:
                    self.best_score = f1_macro
                    self.best_model = model
                    print(f"  *** NEW BEST MODEL ***")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                continue
        
        self.models = results
        return results
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """Perform cross-validation on models"""
        models = self.define_models()
        cv_results = {}
        
        print("\nCross-validation results:")
        print("=" * 60)
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in models.items():
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)
                
                cv_results[name] = {
                    'mean_score': cv_scores.mean(),
                    'std_score': cv_scores.std(),
                    'scores': cv_scores
                }
                
                print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"Error in CV for {name}: {str(e)}")
                continue
        
        return cv_results
    
    def ensemble_predictions(self, X_val, y_val):
        """Create ensemble predictions"""
        if not self.models:
            print("No models trained yet!")
            return None
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, result in self.models.items():
            model = result['model']
            predictions[name] = model.predict(X_val)
            probabilities[name] = model.predict_proba(X_val)
        
        # Majority voting ensemble
        ensemble_pred = np.zeros(len(X_val), dtype=int)
        
        # Convert string predictions to integers for voting
        label_mapping = {label: i for i, label in enumerate(np.unique(y_val))}
        
        for i in range(len(X_val)):
            votes = {}
            for name, pred in predictions.items():
                vote_label = pred[i]
                vote_int = label_mapping[vote_label]
                votes[vote_int] = votes.get(vote_int, 0) + 1
            
            ensemble_pred[i] = max(votes.keys(), key=lambda k: votes[k])
        
        # Convert back to original labels
        inverse_mapping = {i: label for label, i in label_mapping.items()}
        ensemble_pred_labels = [inverse_mapping[i] for i in ensemble_pred]
        
        # Calculate ensemble F1 score
        ensemble_f1 = f1_score(y_val, ensemble_pred_labels, average='macro')
        
        print(f"\nEnsemble F1 Macro: {ensemble_f1:.4f}")
        
        return ensemble_pred_labels, ensemble_f1
    
    def weighted_ensemble_predictions(self, X_val, y_val):
        """Create weighted ensemble based on individual model performance"""
        if not self.models:
            print("No models trained yet!")
            return None
        
        # Get probabilities and weights based on validation performance
        probabilities = {}
        weights = {}
        
        for name, result in self.models.items():
            model = result['model']
            probabilities[name] = model.predict_proba(X_val)
            weights[name] = result['f1_macro']  # Use F1 macro as weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {name: w/total_weight for name, w in weights.items()}
        
        # Weighted average of probabilities
        avg_prob = np.zeros_like(probabilities[list(probabilities.keys())[0]])
        
        for name, prob in probabilities.items():
            avg_prob += weights[name] * prob
        
        # Get final predictions
        ensemble_pred = np.argmax(avg_prob, axis=1)
        
        # Convert to original labels
        classes = list(self.best_model.classes_)
        ensemble_pred_labels = [classes[i] for i in ensemble_pred]
        
        # Calculate ensemble F1 score
        ensemble_f1 = f1_score(y_val, ensemble_pred_labels, average='macro')
        
        print(f"Weighted Ensemble F1 Macro: {ensemble_f1:.4f}")
        
        return ensemble_pred_labels, ensemble_f1, avg_prob
    
    def train_final_model(self, X, y):
        """Train final model on all data"""
        print("\nTraining final model on all data...")
        
        # Use the best model or create an ensemble
        if self.best_model is not None:
            print(f"Using best model: {type(self.best_model).__name__}")
            final_model = self.best_model
        else:
            # Use XGBoost as default (usually performs well)
            final_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=self.random_state,
                n_jobs=-1
            )
        
        final_model.fit(X, y)
        return final_model
    
    def feature_importance_analysis(self, X, feature_names):
        """Analyze feature importance"""
        if self.best_model is None:
            print("No best model available for feature importance!")
            return None
        
        importance_type = None
        importances = None
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            importance_type = "gain"
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_).mean(axis=0)
            importance_type = "coefficient"
        
        if importances is not None:
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nTop 15 Most Important Features:")
            print(feature_importance.head(15).to_string(index=False))
            
            return feature_importance
        
        return None

# Training pipeline
def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('Train.csv')
    test_df = pd.read_csv('Test.csv')
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Feature engineering
    print("Performing feature engineering...")
    train_processed = trainer.feature_engineer.fit_transform(train_df)
    
    # Prepare data
    X = train_processed.drop(['ID', 'Target'], axis=1)
    y = train_processed['Target']
    
    # Split for validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Train and evaluate models
    results = trainer.train_and_evaluate_models(X_train, y_train, X_val, y_val)
    
    # Cross-validation
    cv_results = trainer.cross_validate_models(X, y, cv_folds=5)
    
    # Ensemble predictions
    ensemble_pred, ensemble_f1 = trainer.ensemble_predictions(X_val, y_val)
    
    # Weighted ensemble
    weighted_pred, weighted_f1, weighted_probs = trainer.weighted_ensemble_predictions(X_val, y_val)
    
    # Feature importance
    feature_importance = trainer.feature_importance_analysis(X, X.columns.tolist())
    
    # Train final model
    final_model = trainer.train_final_model(X, y)
    
    print(f"\nBest validation F1 Macro: {trainer.best_score:.4f}")
    print(f"Ensemble F1 Macro: {ensemble_f1:.4f}")
    print(f"Weighted Ensemble F1 Macro: {weighted_f1:.4f}")
    
    return trainer, final_model

if __name__ == "__main__":
    trainer, final_model = main()
