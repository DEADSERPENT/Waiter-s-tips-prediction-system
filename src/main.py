"""
Main Pipeline — Complete End-to-End Execution
Runs the entire tips prediction system pipeline.
Models: CART Tree, Random Forest
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from download_data import download_tips_dataset
from data_preprocessing import DataPreprocessor
from model_training import TipPredictor
from visualization import TipsVisualizer
from prediction import TipPredictionSystem
from explainability import TipExplainer


def run_complete_pipeline():
    print("\n" + "="*70)
    print(" "*15 + "WAITER'S TIPS PREDICTION SYSTEM")
    print(" "*20 + "Complete Pipeline Execution")
    print("="*70)

    # Step 1: Dataset
    print("\n" + "="*70)
    print("STEP 1: DATASET (1,000 rows)")
    print("="*70)
    df = download_tips_dataset()
    if df is None:
        print("[ERR] Failed to load dataset. Exiting.")
        return

    # Step 2: Preprocessing
    print("\n" + "="*70)
    print("STEP 2: DATA PREPROCESSING")
    print("="*70)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _, _ = preprocessor.preprocess_pipeline()

    # Step 3: EDA Visualizations
    print("\n" + "="*70)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    visualizer = TipsVisualizer()
    visualizer.plot_data_distribution(preprocessor.df)
    visualizer.plot_categorical_analysis(preprocessor.df)
    visualizer.plot_correlation_heatmap(preprocessor.df)
    visualizer.plot_scatter_relationships(preprocessor.df)
    print("[OK] EDA visualizations complete")

    # Step 4: Model Training (CART, Random Forest)
    print("\n" + "="*70)
    print("STEP 4: MODEL TRAINING — CART, RANDOM FOREST")
    print("="*70)
    predictor = TipPredictor()
    results   = predictor.train_all_models(X_train, X_test, y_train, y_test)

    comparison_df = predictor.display_comparison()
    feature_names = ['total_bill', 'sex', 'smoker', 'day', 'time', 'size']
    predictor.get_feature_importance(feature_names)
    predictor.save_all_models()

    # Step 5: Evaluation Visualizations
    print("\n" + "="*70)
    print("STEP 5: MODEL EVALUATION VISUALIZATIONS")
    print("="*70)
    visualizer.plot_model_comparison(results)
    best_name = predictor.best_model_name
    visualizer.plot_predictions_vs_actual(y_test, results, best_name)
    visualizer.plot_residuals(y_test, results, best_name)
    visualizer.plot_feature_importance(predictor.best_model, feature_names, best_name)
    print("[OK] Evaluation visualizations complete")

    # Step 6: Demo Predictions
    print("\n" + "="*70)
    print("STEP 6: DEMO PREDICTIONS")
    print("="*70)
    pred_system = TipPredictionSystem()
    pred_system.load_model(best_name.replace(' ', '_').lower())
    pred_system.demo_predictions()

    # Step 7: Feature Importance
    print("\n" + "="*70)
    print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    explainer = TipExplainer(
        model         = predictor.best_model,
        X_train       = X_train,
        X_test        = X_test,
        feature_names = feature_names,
    )
    explainer.run_all()

    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Dataset         : {len(preprocessor.df)} records")
    print(f"  Models Trained  : {len(results)} (CART Tree, Random Forest)")
    print(f"  Best Model      : {best_name}")
    print(f"  Best R² Score   : {results[best_name]['R2_Score']:.4f}")
    print(f"  Best RMSE       : ${results[best_name]['RMSE']:.2f}")

    base_dir = os.path.dirname(os.path.dirname(__file__))
    print(f"\nOutput Locations:")
    print(f"  Models          : {os.path.join(base_dir, 'models')}")
    print(f"  Visualizations  : {os.path.join(base_dir, 'results')}")

    print("\n" + "="*70)
    print("Thank you for using the Waiter's Tips Prediction System!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_complete_pipeline()
