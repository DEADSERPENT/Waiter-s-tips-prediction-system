"""
Main Pipeline - Complete End-to-End Execution
Runs the entire tips prediction system pipeline
(includes hyperparameter tuning and explainability)
"""

import os
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))

from download_data import download_tips_dataset
from data_preprocessing import DataPreprocessor
from model_training import TipPredictor
from visualization import TipsVisualizer
from prediction import TipPredictionSystem
from explainability import TipExplainer

def run_complete_pipeline():
    """
    Execute the complete pipeline:
    1. Download data
    2. Preprocess data
    3. Create visualizations
    4. Train models
    5. Evaluate models
    6. Make predictions
    """
    
    print("\n" + "="*70)
    print(" "*15 + "WAITER'S TIPS PREDICTION SYSTEM")
    print(" "*20 + "Complete Pipeline Execution")
    print("="*70)
    
    # Step 1: Download Dataset
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING DATASET")
    print("="*70)
    df = download_tips_dataset()
    
    if df is None:
        print("✗ Failed to download dataset. Exiting.")
        return
    
    # Step 2: Data Preprocessing
    print("\n" + "="*70)
    print("STEP 2: DATA PREPROCESSING")
    print("="*70)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = preprocessor.preprocess_pipeline()
    
    # Step 3: Exploratory Data Analysis & Visualization
    print("\n" + "="*70)
    print("STEP 3: EXPLORATORY DATA ANALYSIS & VISUALIZATION")
    print("="*70)
    visualizer = TipsVisualizer()
    
    print("\nGenerating visualizations...")
    visualizer.plot_data_distribution(preprocessor.df)
    visualizer.plot_categorical_analysis(preprocessor.df)
    visualizer.plot_correlation_heatmap(preprocessor.df)
    visualizer.plot_scatter_relationships(preprocessor.df)
    
    print("\n✓ EDA visualizations complete!")
    
    # Step 4: Model Training (with GridSearchCV hyperparameter tuning)
    print("\n" + "="*70)
    print("STEP 4: MODEL TRAINING & EVALUATION (with Hyperparameter Tuning)")
    print("="*70)
    predictor = TipPredictor()
    results = predictor.train_all_models(X_train, X_test, y_train, y_test,
                                         tune=True, cv=5)
    
    # Display comparison
    comparison_df = predictor.display_comparison()
    
    # Get feature importance
    feature_names = ['total_bill', 'sex', 'smoker', 'day', 'time', 'size']
    predictor.get_feature_importance(feature_names)
    
    # Save models
    predictor.save_all_models()
    
    # Step 5: Model Evaluation Visualizations
    print("\n" + "="*70)
    print("STEP 5: MODEL EVALUATION VISUALIZATIONS")
    print("="*70)
    
    print("\nGenerating model evaluation plots...")
    visualizer.plot_model_comparison(results)
    
    # Plot for best model
    best_model_name = predictor.best_model_name
    visualizer.plot_predictions_vs_actual(y_test, results, best_model_name)
    visualizer.plot_residuals(y_test, results, best_model_name)
    
    # Feature importance for tree-based models
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
        visualizer.plot_feature_importance(predictor.best_model, feature_names, best_model_name)
    
    print("\n✓ Model evaluation visualizations complete!")
    
    # Step 6: Demo Predictions
    print("\n" + "="*70)
    print("STEP 6: DEMO PREDICTIONS")
    print("="*70)

    prediction_system = TipPredictionSystem()
    prediction_system.load_model(best_model_name.replace(' ', '_').lower())
    prediction_system.demo_predictions()

    # Step 7: Explainability (SHAP + LIME)
    print("\n" + "="*70)
    print("STEP 7: EXPLAINABILITY ANALYSIS (SHAP + LIME)")
    print("="*70)
    try:
        feature_names_xai = ['total_bill', 'sex', 'smoker', 'day', 'time', 'size']
        explainer = TipExplainer(
            model         = predictor.best_model,
            X_train       = X_train,
            X_test        = X_test,
            feature_names = feature_names_xai,
        )
        explainer.run_all(num_lime_samples_to_explain=3)
        print("\n✓ Explainability analysis complete!")
    except ImportError as ie:
        print(f"\n[Skipping explainability] Missing package: {ie}")
        print("  Run: pip install shap lime")
    except Exception as ex:
        print(f"\n[Explainability error] {ex}")

    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE!")
    print("="*70)
    
    print("\n📊 Summary:")
    print(f"  ✓ Dataset: {len(preprocessor.df)} records")
    print(f"  ✓ Models Trained: {len(results)}")
    print(f"  ✓ Best Model: {best_model_name}")
    print(f"  ✓ Best R² Score: {results[best_model_name]['R2_Score']:.4f}")
    print(f"  ✓ Best RMSE: ${results[best_model_name]['RMSE']:.2f}")
    
    print("\n📁 Output Locations:")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    print(f"  • Dataset: {os.path.join(base_dir, 'data', 'tips.csv')}")
    print(f"  • Models: {os.path.join(base_dir, 'models')}")
    print(f"  • Visualizations: {os.path.join(base_dir, 'results')}")
    
    print("\n🎯 Next Steps:")
    print("  1. Review visualizations in the 'results' folder")
    print("  2. Check SHAP + LIME plots in 'results/explainability'")
    print("  3. Use prediction.py for interactive predictions")
    print("  4. Launch the web app:  streamlit run app.py")
    print("  5. Explore the Jupyter notebook for detailed analysis")
    
    print("\n" + "="*70)
    print("Thank you for using the Waiter's Tips Prediction System!")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_complete_pipeline()
