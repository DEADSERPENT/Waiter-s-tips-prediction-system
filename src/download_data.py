"""
Download the Tips dataset from Kaggle or use seaborn's built-in dataset
"""

import os
import pandas as pd
import seaborn as sns

def download_tips_dataset():
    """
    Download the tips dataset. 
    First tries to use seaborn's built-in dataset (which is the same as Kaggle's).
    This is the classic restaurant tips dataset used in data science.
    """
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'tips.csv')
    
    try:
        # Load the tips dataset from seaborn (same as Kaggle dataset)
        print("Downloading tips dataset...")
        tips_df = sns.load_dataset('tips')
        
        # Save to CSV
        tips_df.to_csv(output_path, index=False)
        print(f"✓ Dataset downloaded successfully to: {output_path}")
        
        # Display dataset info
        print("\n" + "="*60)
        print("DATASET INFORMATION")
        print("="*60)
        print(f"Total Records: {len(tips_df)}")
        print(f"Total Features: {len(tips_df.columns)}")
        print(f"\nColumns: {list(tips_df.columns)}")
        print(f"\nDataset Shape: {tips_df.shape}")
        
        print("\n" + "="*60)
        print("FIRST 5 ROWS")
        print("="*60)
        print(tips_df.head())
        
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(tips_df.describe())
        
        print("\n" + "="*60)
        print("DATA TYPES")
        print("="*60)
        print(tips_df.dtypes)
        
        print("\n" + "="*60)
        print("MISSING VALUES")
        print("="*60)
        print(tips_df.isnull().sum())
        
        return tips_df
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nAlternative: You can manually download from:")
        print("https://www.kaggle.com/datasets/jsphyg/tipping")
        print(f"and place it in: {output_path}")
        return None

if __name__ == "__main__":
    download_tips_dataset()
