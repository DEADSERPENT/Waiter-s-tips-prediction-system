"""
Model Training Module
Trains multiple regression models, performs GridSearchCV hyperparameter tuning,
and evaluates performance.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data_preprocessing import DataPreprocessor

class TipPredictor:
    """
    Trains and evaluates multiple regression models for tip prediction.
    Includes GridSearchCV hyperparameter tuning for all applicable models.
    """

    def __init__(self):
        """Initialize the predictor"""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.tuned_params = {}          # stores best params found by GridSearchCV

        # Create models directory
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(self.models_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Model definitions with default (baseline) hyperparameters
    # ------------------------------------------------------------------
    def initialize_models(self):
        """Initialize regression models with baseline hyperparameters."""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression':  Ridge(alpha=1.0),
            'Lasso Regression':  Lasso(alpha=0.1),
            'Decision Tree':     DecisionTreeRegressor(random_state=42, max_depth=5),
            'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
        }
        print(f"✓ Initialized {len(self.models)} models")
        return self.models

    # ------------------------------------------------------------------
    # Hyperparameter grids for GridSearchCV
    # ------------------------------------------------------------------
    def _get_param_grids(self):
        """Return hyperparameter search grids for each tunable model."""
        return {
            'Ridge Regression': {
                'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
            },
            'Lasso Regression': {
                'alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            },
            'Decision Tree': {
                'max_depth':        [3, 4, 5, 6, 8, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf':  [1, 2, 4],
            },
            'Random Forest': {
                'n_estimators':     [50, 100, 200],
                'max_depth':        [5, 8, 10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf':  [1, 2],
            },
            'Gradient Boosting': {
                'n_estimators':  [50, 100, 200],
                'max_depth':     [3, 4, 5],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample':     [0.8, 1.0],
            },
        }

    # ------------------------------------------------------------------
    # GridSearchCV tuning
    # ------------------------------------------------------------------
    def tune_hyperparameters(self, X_train, y_train, cv=5):
        """
        Run GridSearchCV for each tunable model and replace the baseline
        model with the best estimator found.

        Parameters
        ----------
        X_train : array-like
            Scaled training features.
        y_train : array-like
            Training target values.
        cv : int
            Number of cross-validation folds (default 5).
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING (GridSearchCV)")
        print("="*60)

        param_grids = self._get_param_grids()

        for model_name, param_grid in param_grids.items():
            if model_name not in self.models:
                continue

            print(f"\n[{model_name}] Searching over {param_grid} ...")
            base_model = self.models[model_name]

            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring='r2',
                cv=cv,
                n_jobs=-1,
                verbose=0,
                refit=True,
            )
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            best_cv_r2  = grid_search.best_score_

            self.models[model_name]   = grid_search.best_estimator_
            self.tuned_params[model_name] = best_params

            print(f"  Best Params : {best_params}")
            print(f"  Best CV R²  : {best_cv_r2:.4f}")
            print(f"✓ {model_name} tuned successfully")

        print("\n✓ Hyperparameter tuning complete for all models")

    # ------------------------------------------------------------------
    # Training & evaluation helpers
    # ------------------------------------------------------------------
    def train_model(self, model_name, model, X_train, y_train):
        """Train a single model (used after tuning)."""
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        print(f"✓ {model_name} trained successfully")
        return model

    def evaluate_model(self, model_name, model, X_test, y_test):
        """Evaluate model performance and store results."""
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        self.results[model_name] = {
            'MAE':         mae,
            'MSE':         mse,
            'RMSE':        rmse,
            'R2_Score':    r2,
            'predictions': y_pred,
        }

        tuned_note = " (tuned)" if model_name in self.tuned_params else ""
        print(f"\n{model_name}{tuned_note} Performance:")
        print(f"  MAE:      ${mae:.4f}")
        print(f"  MSE:      {mse:.4f}")
        print(f"  RMSE:     ${rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")

        return self.results[model_name]

    # ------------------------------------------------------------------
    # Main training pipeline
    # ------------------------------------------------------------------
    def train_all_models(self, X_train, X_test, y_train, y_test,
                         tune=True, cv=5):
        """
        Initialize models, optionally tune hyperparameters via GridSearchCV,
        then train and evaluate every model.

        Parameters
        ----------
        tune : bool
            If True (default), run GridSearchCV before final training.
        cv   : int
            Cross-validation folds used during tuning.
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)

        self.initialize_models()

        # --- Step 1: Hyperparameter tuning (optional) ---
        if tune:
            self.tune_hyperparameters(X_train, y_train, cv=cv)
        else:
            print("\n[Skipping hyperparameter tuning — using baseline params]")

        # --- Step 2: Final training + evaluation ---
        print("\n" + "="*60)
        print("FINAL MODEL TRAINING & EVALUATION")
        print("="*60)

        for model_name, model in self.models.items():
            trained_model = self.train_model(model_name, model, X_train, y_train)
            self.evaluate_model(model_name, trained_model, X_test, y_test)

        # --- Step 3: Cross-validation scores for best transparency ---
        self._print_cv_summary(X_train, y_train, cv=cv)

        # --- Step 4: Find best model ---
        self.find_best_model()

        return self.results

    def _print_cv_summary(self, X_train, y_train, cv=5):
        """Print cross-validated R² for every model (train-set only)."""
        print("\n" + "="*60)
        print(f"CROSS-VALIDATION SUMMARY ({cv}-Fold, R² Score)")
        print("="*60)
        for model_name, model in self.models.items():
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=cv, scoring='r2', n_jobs=-1)
            print(f"  {model_name:<22}: "
                  f"mean={cv_scores.mean():.4f}  std=±{cv_scores.std():.4f}")

    # ------------------------------------------------------------------
    # Best-model selection
    # ------------------------------------------------------------------
    def find_best_model(self):
        """Identify the best model by test-set R² score."""
        best_r2 = -float('inf')

        for model_name, metrics in self.results.items():
            if metrics['R2_Score'] > best_r2:
                best_r2 = metrics['R2_Score']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]

        print("\n" + "="*60)
        print("BEST MODEL (after tuning)")
        print("="*60)
        print(f"  Model    : {self.best_model_name}")
        print(f"  R² Score : {self.results[self.best_model_name]['R2_Score']:.4f}")
        print(f"  RMSE     : ${self.results[self.best_model_name]['RMSE']:.4f}")
        if self.best_model_name in self.tuned_params:
            print(f"  Best Params: {self.tuned_params[self.best_model_name]}")

        return self.best_model_name, self.best_model

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, model_name=None):
        """Save a trained model to disk."""
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]

        model_path = os.path.join(
            self.models_dir,
            f'{model_name.replace(" ", "_").lower()}.pkl'
        )
        joblib.dump(model, model_path)
        print(f"\n✓ Model saved: {model_path}")
        return model_path

    def save_all_models(self):
        """Save all trained models."""
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        for model_name in self.models.keys():
            self.save_model(model_name)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def display_comparison(self):
        """Display a sorted comparison table of all models."""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df[['MAE', 'MSE', 'RMSE', 'R2_Score']]
        comparison_df = comparison_df.sort_values('R2_Score', ascending=False)
        print("\n", comparison_df.to_string())

        if self.tuned_params:
            print("\nBest Hyperparameters Found:")
            for model_name, params in self.tuned_params.items():
                print(f"  {model_name}: {params}")

        return comparison_df

    def get_feature_importance(self, feature_names):
        """Print feature importance for tree-based models."""
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
                        'Feature':    feature_names,
                        'Importance': importance,
                    }).sort_values('Importance', ascending=False)

                    print(f"\n{model_name}:")
                    print(feature_imp_df.to_string(index=False))


# ----------------------------------------------------------------------
# Standalone entry-point
# ----------------------------------------------------------------------
def main():
    """Main training pipeline (with tuning enabled by default)."""
    print("\n" + "="*60)
    print("WAITER'S TIPS PREDICTION SYSTEM - MODEL TRAINING")
    print("="*60)

    # Preprocess data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = \
        preprocessor.preprocess_pipeline()

    # Train + tune models
    predictor = TipPredictor()
    predictor.train_all_models(X_train, X_test, y_train, y_test, tune=True, cv=5)

    # Display comparison (includes best params)
    predictor.display_comparison()

    # Feature importance
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
