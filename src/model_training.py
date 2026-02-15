"""
Model Training Module
Trains multiple regression models and evaluates performance
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data_preprocessing import DataPreprocessor

class TipPredictor:
    """
    Trains and evaluates multiple regression models for tip prediction
    """
    
    def __init__(self):
        """Initialize the predictor"""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Create models directory
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
    
    def initialize_models(self):
        """Initialize different regression models"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        }
        
        print(f"✓ Initialized {len(self.models)} models")
        return self.models
    
    def train_model(self, model_name, model, X_train, y_train):
        """Train a single model"""
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        print(f"✓ {model_name} trained successfully")
        return model
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """Evaluate model performance"""
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        self.results[model_name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2_Score': r2,
            'predictions': y_pred
        }
        
        print(f"\n{model_name} Performance:")
        print(f"  MAE:  ${mae:.2f}")
        print(f"  MSE:  {mse:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  R² Score: {r2:.4f}")
        
        return self.results[model_name]
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        self.initialize_models()
        
        for model_name, model in self.models.items():
            # Train
            trained_model = self.train_model(model_name, model, X_train, y_train)
            
            # Evaluate
            self.evaluate_model(model_name, trained_model, X_test, y_test)
        
        # Find best model
        self.find_best_model()
        
        return self.results
    
    def find_best_model(self):
        """Find the best performing model based on R2 score"""
        best_r2 = -float('inf')
        
        for model_name, metrics in self.results.items():
            if metrics['R2_Score'] > best_r2:
                best_r2 = metrics['R2_Score']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        print("\n" + "="*60)
        print("BEST MODEL")
        print("="*60)
        print(f"Model: {self.best_model_name}")
        print(f"R² Score: {self.results[self.best_model_name]['R2_Score']:.4f}")
        print(f"RMSE: ${self.results[self.best_model_name]['RMSE']:.2f}")
        
        return self.best_model_name, self.best_model
    
    def save_model(self, model_name=None):
        """Save the trained model"""
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        model_path = os.path.join(self.models_dir, f'{model_name.replace(" ", "_").lower()}.pkl')
        joblib.dump(model, model_path)
        print(f"\n✓ Model saved: {model_path}")
        
        return model_path
    
    def save_all_models(self):
        """Save all trained models"""
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        for model_name in self.models.keys():
            self.save_model(model_name)
    
    def display_comparison(self):
        """Display comparison of all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df[['MAE', 'MSE', 'RMSE', 'R2_Score']]
        comparison_df = comparison_df.sort_values('R2_Score', ascending=False)
        
        print("\n", comparison_df.to_string())
        
        return comparison_df
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from tree-based models"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        
        tree_models = ['Decision Tree', 'Random Forest', 'Gradient Boosting']
        
        for model_name in tree_models:
            if model_name in self.models:
                model = self.models[model_name]
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    feature_imp_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False)
                    
                    print(f"\n{model_name}:")
                    print(feature_imp_df.to_string(index=False))

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("WAITER'S TIPS PREDICTION SYSTEM - MODEL TRAINING")
    print("="*60)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = preprocessor.preprocess_pipeline()
    
    # Train models
    predictor = TipPredictor()
    predictor.train_all_models(X_train, X_test, y_train, y_test)
    
    # Display comparison
    predictor.display_comparison()
    
    # Get feature importance
    feature_names = ['total_bill', 'sex', 'smoker', 'day', 'time', 'size']
    predictor.get_feature_importance(feature_names)
    
    # Save models
    predictor.save_all_models()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    return predictor

if __name__ == "__main__":
    predictor = main()
