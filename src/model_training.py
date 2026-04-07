"""
Model Training Module
Trains two decision-tree-based models for tip prediction:
  - CART Tree    (criterion='squared_error')
  - Random Forest
Best hyperparameters are pre-tuned and hard-coded for the 1,000-row dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data_preprocessing import DataPreprocessor


class TipPredictor:
    """
    Trains and evaluates CART Tree and Random Forest
    regressors for waiter tip prediction.
    """

    def __init__(self):
        self.models          = {}
        self.results         = {}
        self.best_model      = None
        self.best_model_name = None
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(self.models_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def initialize_models(self):
        """
        Two models with best-known hyperparameters for a ~1,000-row
        tips dataset with 6 features.

        CART Tree     — standard MSE split (Classification and Regression Tree).
        Random Forest — ensemble of 200 CART trees with feature sub-sampling.
        """
        self.models = {
            'CART Tree': DecisionTreeRegressor(
                criterion        = 'squared_error',
                max_depth        = 5,
                min_samples_split= 10,
                min_samples_leaf = 4,
                random_state     = 42,
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators     = 200,
                max_depth        = 8,
                min_samples_split= 10,
                min_samples_leaf = 3,
                max_features     = 'sqrt',
                n_jobs           = -1,
                random_state     = 42,
            ),
        }
        print(f"[OK] Initialized {len(self.models)} models: {list(self.models.keys())}")
        return self.models

    # ------------------------------------------------------------------
    def train_model(self, model_name, model, X_train, y_train):
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        print(f"[OK] {model_name} trained")
        return model

    def evaluate_model(self, model_name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        mse    = mean_squared_error(y_test, y_pred)
        rmse   = np.sqrt(mse)
        r2     = r2_score(y_test, y_pred)

        self.results[model_name] = {
            'MAE': mae, 'MSE': mse, 'RMSE': rmse,
            'R2_Score': r2, 'predictions': y_pred,
        }

        print(f"\n{model_name} Performance:")
        print(f"  MAE:      ${mae:.4f}")
        print(f"  RMSE:     ${rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")
        return self.results[model_name]

    # ------------------------------------------------------------------
    def train_all_models(self, X_train, X_test, y_train, y_test,
                         tune=False, cv=5):
        """Train and evaluate all three models."""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)

        self.initialize_models()

        print("\n" + "="*60)
        print("MODEL TRAINING & EVALUATION")
        print("="*60)

        for model_name, model in self.models.items():
            trained = self.train_model(model_name, model, X_train, y_train)
            self.evaluate_model(model_name, trained, X_test, y_test)

        self._print_cv_summary(X_train, y_train, cv=cv)
        self.find_best_model()
        return self.results

    def _print_cv_summary(self, X_train, y_train, cv=5):
        print("\n" + "="*60)
        print(f"CROSS-VALIDATION ({cv}-Fold R² Score)")
        print("="*60)
        for name, model in self.models.items():
            scores = cross_val_score(model, X_train, y_train,
                                     cv=cv, scoring='r2', n_jobs=-1)
            print(f"  {name:<16}: mean={scores.mean():.4f}  std=±{scores.std():.4f}")

    # ------------------------------------------------------------------
    def find_best_model(self):
        best_r2 = -float('inf')
        for name, metrics in self.results.items():
            if metrics['R2_Score'] > best_r2:
                best_r2 = metrics['R2_Score']
                self.best_model_name = name
                self.best_model      = self.models[name]

        print("\n" + "="*60)
        print("BEST MODEL")
        print("="*60)
        print(f"  Model    : {self.best_model_name}")
        print(f"  R² Score : {self.results[self.best_model_name]['R2_Score']:.4f}")
        print(f"  RMSE     : ${self.results[self.best_model_name]['RMSE']:.4f}")
        return self.best_model_name, self.best_model

    # ------------------------------------------------------------------
    def save_model(self, model_name=None):
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        path = os.path.join(self.models_dir,
                            f'{model_name.replace(" ", "_").lower()}.pkl')
        joblib.dump(model, path)
        print(f"[OK] Saved: {path}")
        return path

    def save_all_models(self):
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        for name in self.models:
            self.save_model(name)

    # ------------------------------------------------------------------
    def display_comparison(self):
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        df = pd.DataFrame(self.results).T[['MAE', 'MSE', 'RMSE', 'R2_Score']]
        df = df.sort_values('R2_Score', ascending=False)
        print(df.to_string())
        return df

    def get_feature_importance(self, feature_names):
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                imp_df = pd.DataFrame({
                    'Feature':    feature_names,
                    'Importance': model.feature_importances_,
                }).sort_values('Importance', ascending=False)
                print(f"\n{name}:")
                print(imp_df.to_string(index=False))


# ----------------------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("WAITER'S TIPS PREDICTION — MODEL TRAINING")
    print("="*60)

    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _, _ = preprocessor.preprocess_pipeline()

    predictor = TipPredictor()
    predictor.train_all_models(X_train, X_test, y_train, y_test)
    predictor.display_comparison()

    feature_names = ['total_bill', 'sex', 'smoker', 'day', 'time', 'size']
    predictor.get_feature_importance(feature_names)
    predictor.save_all_models()

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    return predictor


if __name__ == "__main__":
    main()
