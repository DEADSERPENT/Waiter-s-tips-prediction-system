"""
Download the Tips dataset from Kaggle or use seaborn's built-in dataset
"""

import os
import pandas as pd
import seaborn as sns

def download_tips_dataset():
    """
    Load the tips dataset from data/tips.csv.
    If the file already exists it is used as-is (preserving any custom dataset).
    Only falls back to seaborn if the file is missing entirely.
    """

    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, 'tips.csv')

    try:
        # Use existing file if present — never overwrite a custom dataset
        if os.path.exists(output_path):
            print(f"✓ Dataset found at: {output_path} — using existing file")
            tips_df = pd.read_csv(output_path)
        else:
            print("Dataset not found locally. Loading from seaborn as fallback...")
            tips_df = sns.load_dataset('tips')
            tips_df.to_csv(output_path, index=False)
            print(f"✓ Dataset saved to: {output_path}")

        print(f"✓ Dataset loaded successfully: {tips_df.shape}")
        
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
