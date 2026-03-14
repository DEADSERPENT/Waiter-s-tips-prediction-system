"""
Explainability Module
Provides model explanations using SHAP (SHapley Additive exPlanations)
and LIME (Local Interpretable Model-agnostic Explanations).

Usage (standalone):
    python src/explainability.py

Usage (from main pipeline):
    from src.explainability import TipExplainer
    explainer = TipExplainer(model, X_train, X_test, feature_names)
    explainer.run_all()
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# ──────────────────────────────────────────────────────────────────────────────
# Helper: safe imports with friendly error messages
# ──────────────────────────────────────────────────────────────────────────────
def _require(package_name, install_name=None):
    """Import a package and raise a helpful error if not installed."""
    import importlib
    try:
        return importlib.import_module(package_name)
    except ImportError:
        pip_name = install_name or package_name
        raise ImportError(
            f"\n[Missing package] '{package_name}' is not installed.\n"
            f"Run:  pip install {pip_name}\n"
        )


# ──────────────────────────────────────────────────────────────────────────────
class TipExplainer:
    """
    Generates SHAP and LIME explanations for any trained scikit-learn
    regression model used in the Waiter's Tips Prediction System.

    Parameters
    ----------
    model        : trained sklearn estimator
    X_train      : np.ndarray — scaled training features
    X_test       : np.ndarray — scaled test features
    feature_names: list[str]  — names matching the feature columns
    results_dir  : str        — folder where plots are saved
    """

    FEATURE_NAMES_DEFAULT = ['total_bill', 'sex', 'smoker', 'day', 'time', 'size']

    def __init__(self, model, X_train, X_test,
                 feature_names=None, results_dir=None):
        self.model         = model
        self.X_train       = np.array(X_train)
        self.X_test        = np.array(X_test)
        self.feature_names = feature_names or self.FEATURE_NAMES_DEFAULT

        if results_dir is None:
            base = os.path.dirname(os.path.dirname(__file__))
            results_dir = os.path.join(base, 'results', 'explainability')
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir

        print(f"\n✓ TipExplainer initialised")
        print(f"  Features : {self.feature_names}")
        print(f"  Train rows: {self.X_train.shape[0]}  |  Test rows: {self.X_test.shape[0]}")
        print(f"  Output dir: {self.results_dir}")

    # ──────────────────────────────────────────────────────────────────
    # SHAP
    # ──────────────────────────────────────────────────────────────────
    def explain_shap(self, max_display=6, num_samples=100):
        """
        Compute SHAP values and save four standard explanation plots:
          1. Summary bar plot  — global feature importance
          2. Beeswarm plot     — global impact distribution
          3. Waterfall plot    — single-prediction breakdown (first test row)
          4. Dependence plot   — total_bill vs SHAP value

        Parameters
        ----------
        max_display  : max features shown in summary plots
        num_samples  : background samples for KernelExplainer fallback
        """
        shap = _require('shap')

        print("\n" + "="*60)
        print("SHAP EXPLANATIONS")
        print("="*60)

        # Choose explainer automatically
        explainer = self._build_shap_explainer(shap, num_samples)

        print("  Computing SHAP values …")
        shap_values = explainer(self.X_test)

        # ── Plot 1: Summary bar (global importance) ──────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.bar(shap_values, max_display=max_display, show=False, ax=ax)
        ax.set_title("SHAP — Global Feature Importance (mean |SHAP value|)",
                     fontsize=13, pad=12)
        path = os.path.join(self.results_dir, 'shap_summary_bar.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved: {path}")

        # ── Plot 2: Beeswarm ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        plt.title("SHAP — Beeswarm (Impact on Model Output)", fontsize=13, pad=12)
        path = os.path.join(self.results_dir, 'shap_beeswarm.png')
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  ✓ Saved: {path}")

        # ── Plot 3: Waterfall for first test sample ───────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.title("SHAP — Waterfall (Single Prediction Explanation, sample 0)",
                  fontsize=12, pad=12)
        path = os.path.join(self.results_dir, 'shap_waterfall_sample0.png')
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  ✓ Saved: {path}")

        # ── Plot 4: Dependence plot — total_bill (index 0) ───────────
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            feat_idx = self.feature_names.index('total_bill')
            shap.plots.scatter(shap_values[:, feat_idx], show=False, ax=ax)
            ax.set_title("SHAP Dependence — total_bill", fontsize=13, pad=12)
            path = os.path.join(self.results_dir, 'shap_dependence_total_bill.png')
            fig.savefig(path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"  ✓ Saved: {path}")
        except Exception as e:
            print(f"  [Warning] Dependence plot skipped: {e}")

        print("\n✓ SHAP analysis complete")
        return shap_values

    def _build_shap_explainer(self, shap, num_samples):
        """Select the fastest appropriate SHAP explainer for the model."""
        model_type = type(self.model).__name__
        tree_types  = ('RandomForest', 'GradientBoosting', 'DecisionTree',
                       'ExtraTree', 'XGB', 'LGB', 'CatBoost')
        linear_types = ('LinearRegression', 'Ridge', 'Lasso', 'ElasticNet')

        if any(t in model_type for t in tree_types):
            print(f"  Using TreeExplainer for {model_type}")
            return shap.TreeExplainer(self.model)
        elif any(t in model_type for t in linear_types):
            print(f"  Using LinearExplainer for {model_type}")
            background = shap.maskers.Independent(self.X_train, max_samples=num_samples)
            return shap.LinearExplainer(self.model, background)
        else:
            print(f"  Using KernelExplainer for {model_type} (slower)")
            background = shap.sample(self.X_train, num_samples)
            return shap.KernelExplainer(self.model.predict, background)

    # ──────────────────────────────────────────────────────────────────
    # LIME
    # ──────────────────────────────────────────────────────────────────
    def explain_lime(self, num_samples_to_explain=3, num_lime_samples=1000):
        """
        Generate LIME local explanations for the first N test instances.
        Saves one bar-chart PNG per instance.

        Parameters
        ----------
        num_samples_to_explain : how many test rows to explain
        num_lime_samples       : perturbation samples LIME uses internally
        """
        lime_tabular = _require('lime.lime_tabular', install_name='lime')

        print("\n" + "="*60)
        print("LIME EXPLANATIONS")
        print("="*60)

        # Build LIME explainer on training distribution
        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data   = self.X_train,
            feature_names   = self.feature_names,
            mode            = 'regression',
            discretize_continuous = True,
            random_state    = 42,
        )

        n = min(num_samples_to_explain, len(self.X_test))
        for i in range(n):
            instance = self.X_test[i]
            print(f"\n  Explaining test sample {i} …")

            exp = lime_explainer.explain_instance(
                data_row        = instance,
                predict_fn      = self.model.predict,
                num_features    = len(self.feature_names),
                num_samples     = num_lime_samples,
            )

            predicted_val = self.model.predict(instance.reshape(1, -1))[0]
            exp_list = exp.as_list()          # [(feature_label, weight), …]

            # ── Bar chart ────────────────────────────────────────────
            labels  = [x[0] for x in exp_list]
            weights = [x[1] for x in exp_list]
            colors  = ['#2ecc71' if w >= 0 else '#e74c3c' for w in weights]

            fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.55)))
            bars = ax.barh(labels, weights, color=colors, edgecolor='white', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_xlabel("LIME Weight (contribution to prediction)", fontsize=11)
            ax.set_title(
                f"LIME — Local Explanation for Sample {i}\n"
                f"Predicted Tip: ${predicted_val:.2f}",
                fontsize=12, pad=10,
            )
            ax.invert_yaxis()
            plt.tight_layout()

            path = os.path.join(self.results_dir, f'lime_explanation_sample{i}.png')
            fig.savefig(path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"  ✓ Saved: {path}")
            print(f"     Predicted tip: ${predicted_val:.2f}")
            for label, weight in exp_list:
                direction = "↑ increases" if weight >= 0 else "↓ decreases"
                print(f"     {direction} tip  |  {label}  (weight={weight:+.4f})")

        print("\n✓ LIME analysis complete")

    # ──────────────────────────────────────────────────────────────────
    # Global feature-importance comparison (SHAP-based)
    # ──────────────────────────────────────────────────────────────────
    def plot_shap_importance_comparison(self, shap_values):
        """
        Bar chart comparing mean |SHAP| value per feature vs
        the model's built-in feature_importances_ (if available).
        Saved as 'shap_vs_builtin_importance.png'.
        """
        print("\n  Plotting SHAP vs built-in importance comparison …")

        mean_shap = np.abs(shap_values.values).mean(axis=0)
        shap_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean |SHAP|': mean_shap,
        }).sort_values('Mean |SHAP|', ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: SHAP
        axes[0].barh(shap_df['Feature'], shap_df['Mean |SHAP|'],
                     color='#3498db', edgecolor='white')
        axes[0].invert_yaxis()
        axes[0].set_title("Mean |SHAP| Value\n(Global Importance)", fontsize=12)
        axes[0].set_xlabel("Mean |SHAP value|")

        # Right: built-in importance (tree models only)
        if hasattr(self.model, 'feature_importances_'):
            builtin = self.model.feature_importances_
            builtin_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': builtin,
            }).sort_values('Importance', ascending=False)

            axes[1].barh(builtin_df['Feature'], builtin_df['Importance'],
                         color='#e67e22', edgecolor='white')
            axes[1].invert_yaxis()
            axes[1].set_title("Built-in Feature Importance\n(from model)", fontsize=12)
            axes[1].set_xlabel("Importance score")
        else:
            axes[1].text(0.5, 0.5, "Not available\n(linear model)",
                         ha='center', va='center', fontsize=12,
                         transform=axes[1].transAxes)
            axes[1].set_title("Built-in Feature Importance", fontsize=12)

        fig.suptitle("SHAP vs Built-in Feature Importance", fontsize=14, y=1.02)
        plt.tight_layout()

        path = os.path.join(self.results_dir, 'shap_vs_builtin_importance.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved: {path}")

    # ──────────────────────────────────────────────────────────────────
    # Convenience: run everything
    # ──────────────────────────────────────────────────────────────────
    def run_all(self, num_lime_samples_to_explain=3):
        """Run full SHAP + LIME pipeline and save all plots."""
        print("\n" + "="*60)
        print("EXPLAINABILITY ANALYSIS (SHAP + LIME)")
        print("="*60)

        # SHAP
        shap_values = self.explain_shap()

        # SHAP vs built-in comparison
        self.plot_shap_importance_comparison(shap_values)

        # LIME
        self.explain_lime(num_samples_to_explain=num_lime_samples_to_explain)

        print("\n" + "="*60)
        print("EXPLAINABILITY ANALYSIS COMPLETE")
        print(f"All plots saved to: {self.results_dir}")
        print("="*60)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry-point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """
    Full standalone run:
      1. Preprocess the tips dataset
      2. Train the best model (with tuning)
      3. Run SHAP + LIME explanations
    """
    from src.data_preprocessing import DataPreprocessor
    from src.model_training import TipPredictor

    print("\n" + "="*60)
    print("WAITER'S TIPS PREDICTION SYSTEM — EXPLAINABILITY")
    print("="*60)

    # --- Preprocess ---
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _, _ = preprocessor.preprocess_pipeline()

    # --- Train (with tuning) ---
    predictor = TipPredictor()
    predictor.train_all_models(X_train, X_test, y_train, y_test, tune=True, cv=5)
    predictor.save_all_models()

    best_model = predictor.best_model
    print(f"\nUsing best model: {predictor.best_model_name}")

    # --- Explain ---
    feature_names = ['total_bill', 'sex', 'smoker', 'day', 'time', 'size']
    explainer = TipExplainer(
        model         = best_model,
        X_train       = X_train,
        X_test        = X_test,
        feature_names = feature_names,
    )
    explainer.run_all(num_lime_samples_to_explain=3)


if __name__ == "__main__":
    main()
