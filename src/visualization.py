"""
Visualization Module
Creates comprehensive visualizations for data analysis and model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class TipsVisualizer:
    """
    Creates visualizations for tips prediction analysis
    """
    
    def __init__(self):
        """Initialize visualizer"""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        
        # Create results directory
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def plot_data_distribution(self, df):
        """Plot distribution of numerical features"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Distribution of Features', fontsize=16, fontweight='bold')
        
        numerical_cols = ['total_bill', 'tip', 'size']
        
        for idx, col in enumerate(numerical_cols):
            # Histogram
            axes[0, idx].hist(df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            axes[0, idx].set_title(f'{col.replace("_", " ").title()} Distribution')
            axes[0, idx].set_xlabel(col.replace("_", " ").title())
            axes[0, idx].set_ylabel('Frequency')
            
            # Box plot
            axes[1, idx].boxplot(df[col], vert=True)
            axes[1, idx].set_title(f'{col.replace("_", " ").title()} Box Plot')
            axes[1, idx].set_ylabel(col.replace("_", " ").title())
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'data_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_categorical_analysis(self, df):
        """Analyze categorical features"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Categorical Features Analysis', fontsize=16, fontweight='bold')
        
        # Sex vs Tip
        sns.boxplot(data=df, x='sex', y='tip', ax=axes[0, 0], palette='Set2')
        axes[0, 0].set_title('Tips by Gender')
        axes[0, 0].set_xlabel('Gender')
        axes[0, 0].set_ylabel('Tip Amount ($)')
        
        # Smoker vs Tip
        sns.boxplot(data=df, x='smoker', y='tip', ax=axes[0, 1], palette='Set3')
        axes[0, 1].set_title('Tips by Smoking Status')
        axes[0, 1].set_xlabel('Smoker')
        axes[0, 1].set_ylabel('Tip Amount ($)')
        
        # Day vs Tip
        sns.boxplot(data=df, x='day', y='tip', ax=axes[1, 0], palette='husl')
        axes[1, 0].set_title('Tips by Day of Week')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Tip Amount ($)')
        
        # Time vs Tip
        sns.boxplot(data=df, x='time', y='tip', ax=axes[1, 1], palette='pastel')
        axes[1, 1].set_title('Tips by Time of Day')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Tip Amount ($)')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'categorical_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_correlation_heatmap(self, df):
        """Plot correlation heatmap"""
        # Select numerical columns and encoded categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        corr_df = df[numerical_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'correlation_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_scatter_relationships(self, df):
        """Plot scatter plots for key relationships"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Relationship Analysis', fontsize=16, fontweight='bold')
        
        # Total Bill vs Tip
        axes[0].scatter(df['total_bill'], df['tip'], alpha=0.6, c='blue', edgecolors='black')
        axes[0].set_xlabel('Total Bill ($)')
        axes[0].set_ylabel('Tip ($)')
        axes[0].set_title('Total Bill vs Tip')
        axes[0].grid(True, alpha=0.3)
        
        # Add regression line
        z = np.polyfit(df['total_bill'], df['tip'], 1)
        p = np.poly1d(z)
        axes[0].plot(df['total_bill'], p(df['total_bill']), "r--", linewidth=2, label='Trend')
        axes[0].legend()
        
        # Size vs Tip
        axes[1].scatter(df['size'], df['tip'], alpha=0.6, c='green', edgecolors='black')
        axes[1].set_xlabel('Party Size')
        axes[1].set_ylabel('Tip ($)')
        axes[1].set_title('Party Size vs Tip')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'scatter_relationships.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_model_comparison(self, results_dict):
        """Plot model performance comparison"""
        models = list(results_dict.keys())
        r2_scores = [results_dict[m]['R2_Score'] for m in models]
        rmse_scores = [results_dict[m]['RMSE'] for m in models]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # R² Score comparison
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        axes[0].barh(models, r2_scores, color=colors, edgecolor='black')
        axes[0].set_xlabel('R² Score')
        axes[0].set_title('R² Score by Model')
        axes[0].set_xlim(0, 1)
        axes[0].grid(axis='x', alpha=0.3)
        
        # RMSE comparison
        axes[1].barh(models, rmse_scores, color=colors, edgecolor='black')
        axes[1].set_xlabel('RMSE ($)')
        axes[1].set_title('RMSE by Model')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_predictions_vs_actual(self, y_test, predictions_dict, model_name):
        """Plot actual vs predicted values"""
        y_pred = predictions_dict[model_name]['predictions']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', s=50)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Tip ($)', fontsize=12)
        plt.ylabel('Predicted Tip ($)', fontsize=12)
        plt.title(f'Actual vs Predicted Tips - {model_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score to plot
        r2 = predictions_dict[model_name]['R2_Score']
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, f'predictions_vs_actual_{model_name.replace(" ", "_").lower()}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_residuals(self, y_test, predictions_dict, model_name):
        """Plot residual analysis"""
        y_pred = predictions_dict[model_name]['predictions']
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16, fontweight='bold')
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='black')
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Tip ($)')
        axes[0].set_ylabel('Residuals ($)')
        axes[0].set_title('Residuals vs Predicted Values')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals ($)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, f'residuals_{model_name.replace(" ", "_").lower()}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_feature_importance(self, model, feature_names, model_name):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importance)), importance[indices], color='teal', edgecolor='black', alpha=0.7)
            plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(self.results_dir, f'feature_importance_{model_name.replace(" ", "_").lower()}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
            plt.close()

if __name__ == "__main__":
    print("Visualization module - import and use in other scripts")
