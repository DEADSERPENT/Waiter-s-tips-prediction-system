"""
Data Preprocessing Module
Handles data cleaning, encoding, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    """
    Handles all data preprocessing tasks for the tips prediction system
    """
    
    def __init__(self, data_path=None):
        """Initialize the preprocessor"""
        if data_path is None:
            # Default path
            base_dir = os.path.dirname(os.path.dirname(__file__))
            data_path = os.path.join(base_dir, 'data', 'tips.csv')
        
        self.data_path = data_path
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Data loaded successfully: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def explore_data(self):
        """Display basic data exploration"""
        if self.df is None:
            print("✗ No data loaded. Call load_data() first.")
            return
        
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        print("\nDataset Shape:", self.df.shape)
        print("\nColumn Names:", list(self.df.columns))
        print("\nData Types:\n", self.df.dtypes)
        print("\nMissing Values:\n", self.df.isnull().sum())
        print("\nBasic Statistics:\n", self.df.describe())
        
        print("\nCategorical Features Distribution:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts())
    
    def handle_missing_values(self):
        """Handle missing values if any"""
        if self.df is None:
            return
        
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values. Handling...")
            # For numerical columns, fill with median
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())
            
            # For categorical columns, fill with mode
            cat_cols = self.df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            
            print("✓ Missing values handled")
        else:
            print("✓ No missing values found")
    
    def encode_categorical_features(self):
        """Encode categorical variables"""
        if self.df is None:
            return
        
        categorical_cols = ['sex', 'smoker', 'day', 'time']
        
        print("\nEncoding categorical features...")
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                print(f"✓ Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        return self.df
    
    def create_features(self):
        """Create additional features"""
        if self.df is None:
            return
        
        print("\nCreating engineered features...")
        
        # Tip percentage
        self.df['tip_percentage'] = (self.df['tip'] / self.df['total_bill']) * 100
        
        # Bill per person
        self.df['bill_per_person'] = self.df['total_bill'] / self.df['size']
        
        # Tip per person
        self.df['tip_per_person'] = self.df['tip'] / self.df['size']
        
        print("✓ Created features: tip_percentage, bill_per_person, tip_per_person")
        
        return self.df
    
    def prepare_features(self, target_col='tip'):
        """Prepare features and target for modeling"""
        if self.df is None:
            return None, None
        
        # Select features for modeling
        feature_cols = ['total_bill', 'sex_encoded', 'smoker_encoded', 
                       'day_encoded', 'time_encoded', 'size']
        
        X = self.df[feature_cols]
        y = self.df[target_col]
        
        print(f"\n✓ Features prepared: {X.shape}")
        print(f"✓ Target prepared: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n✓ Data split complete:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Testing set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("✓ Features scaled")
        
        return X_train_scaled, X_test_scaled
    
    def preprocess_pipeline(self):
        """Complete preprocessing pipeline"""
        print("\n" + "="*60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Explore data
        self.explore_data()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Encode categorical features
        self.encode_categorical_features()
        
        # Create features
        self.create_features()
        
        # Prepare features and target
        X, y = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = preprocessor.preprocess_pipeline()
