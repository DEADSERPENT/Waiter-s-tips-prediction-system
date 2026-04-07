"""
Feature Importance Module
Displays built-in feature importances for tree-based models.
SHAP and LIME have been removed — only native sklearn importances are used.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

FEATURE_NAMES_DEFAULT = ['total_bill', 'sex', 'smoker', 'day', 'time', 'size']


class TipExplainer:
    """Plots feature importance from trained tree-based models."""

    def __init__(self, model, X_train, X_test,
                 feature_names=None, results_dir=None):
        self.model         = model
        self.X_train       = np.array(X_train)
        self.X_test        = np.array(X_test)
        self.feature_names = feature_names or FEATURE_NAMES_DEFAULT

        if results_dir is None:
            base = os.path.dirname(os.path.dirname(__file__))
            results_dir = os.path.join(base, 'results', 'explainability')
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir

        print(f"\n[OK] TipExplainer ready — output: {self.results_dir}")

    def plot_feature_importance(self):
        """Save a bar chart of built-in feature importances."""
        if not hasattr(self.model, 'feature_importances_'):
            print("  Model has no feature_importances_ — skipping plot.")
            return

        importances = self.model.feature_importances_
        idx = np.argsort(importances)[::-1]
        sorted_names  = [self.feature_names[i] for i in idx]
        sorted_scores = importances[idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#D4AF37' if i == 0 else '#4B5563' for i in range(len(sorted_names))]
        ax.barh(sorted_names[::-1], sorted_scores[::-1],
                color=colors[::-1], edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Feature Importance — {type(self.model).__name__}',
                     fontsize=13, pad=10)
        ax.set_facecolor('#1F2937')
        fig.patch.set_facecolor('#0B0B0B')
        ax.tick_params(colors='#9CA3AF')
        ax.xaxis.label.set_color('#9CA3AF')
        ax.title.set_color('#F9FAFB')
        plt.tight_layout()

        path = os.path.join(self.results_dir, 'feature_importance.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  [OK] Saved: {path}")

        # Print table
        df = pd.DataFrame({'Feature': sorted_names, 'Importance': sorted_scores})
        print(df.to_string(index=False))
        return df

    def run_all(self, **kwargs):
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        self.plot_feature_importance()
        print("[OK] Analysis complete")


# ----------------------------------------------------------------------
def main():
    from src.data_preprocessing import DataPreprocessor
    from src.model_training import TipPredictor

    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _, _ = preprocessor.preprocess_pipeline()

    predictor = TipPredictor()
    predictor.train_all_models(X_train, X_test, y_train, y_test)

    explainer = TipExplainer(
        model         = predictor.best_model,
        X_train       = X_train,
        X_test        = X_test,
        feature_names = ['total_bill', 'sex', 'smoker', 'day', 'time', 'size'],
    )
    explainer.run_all()


if __name__ == "__main__":
    main()
