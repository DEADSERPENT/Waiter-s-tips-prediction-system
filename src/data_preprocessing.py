"""
Data Preprocessing Module
Handles data loading, encoding, and feature preparation.
No feature scaling — tree-based models do not require it.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Preprocessing pipeline for the tips prediction system.
    Features used: total_bill, sex, smoker, day, time, size  (6 features)
    Target: tip
    """

    FEATURE_COLS = ['total_bill', 'sex_encoded', 'smoker_encoded',
                    'day_encoded', 'time_encoded', 'size']
    TARGET_COL   = 'tip'

    def __init__(self, data_path=None):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.data_path     = data_path or os.path.join(base_dir, 'data', 'tips.csv')
        self.df            = None
        self.label_encoders = {}

    # ------------------------------------------------------------------
    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"[OK] Data loaded: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"[ERR] Error loading data: {e}")
            return None

    def explore_data(self):
        if self.df is None:
            return
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        print(f"Shape         : {self.df.shape}")
        print(f"Columns       : {list(self.df.columns)}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nStatistics:\n{self.df.describe()}")

    def handle_missing_values(self):
        if self.df is None:
            return
        missing = self.df.isnull().sum().sum()
        if missing > 0:
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())
            cat_cols = self.df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            print(f"[OK] Handled {missing} missing values")
        else:
            print("[OK] No missing values")

    def encode_categorical_features(self):
        """Label-encode sex, smoker, day, time."""
        if self.df is None:
            return
        for col in ['sex', 'smoker', 'day', 'time']:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                print(f"  Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        print("[OK] Categorical encoding done")

    def prepare_features(self):
        X = self.df[self.FEATURE_COLS].copy()
        y = self.df[self.TARGET_COL].copy()
        print(f"[OK] Features: {X.shape}  |  Target: {y.shape}")
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"[OK] Train: {X_train.shape}  |  Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    def preprocess_pipeline(self):
        """
        Full pipeline: load → clean → encode → split.
        Returns (X_train, X_test, y_train, y_test, X_train, X_test)
        — last two are unscaled copies kept for API compatibility.
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60)

        self.load_data()
        self.explore_data()
        self.handle_missing_values()
        self.encode_categorical_features()

        X, y = self.prepare_features()
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)

        # Return 6 values for API compatibility (no scaling — return X as-is twice)
        return X_train, X_test, y_train, y_test, X_train, X_test


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_pipeline()
