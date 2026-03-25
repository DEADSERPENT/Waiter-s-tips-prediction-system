"""
Prediction Module
Make predictions on new data using trained models
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class TipPredictionSystem:
    """
    System for making tip predictions using trained models
    """
    
    def __init__(self, model_path=None):
        """Initialize prediction system"""
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

        # Load scaler saved during training
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        if model_path:
            self.model = joblib.load(model_path)
            self.model_name = os.path.basename(model_path).replace('.pkl', '').replace('_', ' ').title()
        else:
            self.model = None
            self.model_name = None
    
    def load_model(self, model_name='random_forest'):
        """Load a specific trained model"""
        model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.model_name = model_name.replace('_', ' ').title()
            print(f"✓ Loaded model: {self.model_name}")
            return self.model
        else:
            print(f"✗ Model not found: {model_path}")
            print(f"Available models in {self.models_dir}:")
            if os.path.exists(self.models_dir):
                for file in os.listdir(self.models_dir):
                    if file.endswith('.pkl'):
                        print(f"  - {file}")
            return None
    
    def encode_input(self, sex, smoker, day, time):
        """Encode categorical inputs — must match LabelEncoder alphabetical order."""
        sex_map    = {'Female': 0, 'Male': 1}
        smoker_map = {'No': 0, 'Yes': 1}
        day_map    = {'Fri': 0, 'Sat': 1, 'Sun': 2, 'Thur': 3}
        time_map   = {'Dinner': 0, 'Lunch': 1}
        
        sex_encoded = sex_map.get(sex, 0)
        smoker_encoded = smoker_map.get(smoker, 0)
        day_encoded = day_map.get(day, 0)
        time_encoded = time_map.get(time, 0)
        
        return sex_encoded, smoker_encoded, day_encoded, time_encoded
    
    def predict_tip(self, total_bill, sex, smoker, day, time, size):
        """
        Predict tip amount for given input
        
        Parameters:
        -----------
        total_bill : float
            Total bill amount in dollars
        sex : str
            Gender ('Male' or 'Female')
        smoker : str
            Smoking status ('Yes' or 'No')
        day : str
            Day of week ('Thur', 'Fri', 'Sat', 'Sun')
        time : str
            Time of day ('Lunch' or 'Dinner')
        size : int
            Number of people in party
        
        Returns:
        --------
        float : Predicted tip amount
        """
        
        if self.model is None:
            print("✗ No model loaded. Please load a model first.")
            return None
        
        # Encode categorical features
        sex_enc, smoker_enc, day_enc, time_enc = self.encode_input(sex, smoker, day, time)
        
        # Create feature array and apply scaler if available
        features = np.array([[total_bill, sex_enc, smoker_enc, day_enc, time_enc, size]])
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Make prediction
        prediction = self.model.predict(features)[0]
        
        return prediction
    
    def predict_batch(self, input_df):
        """
        Predict tips for a batch of inputs
        
        Parameters:
        -----------
        input_df : DataFrame
            DataFrame with columns: total_bill, sex, smoker, day, time, size
        
        Returns:
        --------
        array : Predicted tip amounts
        """
        
        if self.model is None:
            print("✗ No model loaded. Please load a model first.")
            return None
        
        # Encode categorical features
        predictions = []
        for _, row in input_df.iterrows():
            pred = self.predict_tip(
                row['total_bill'],
                row['sex'],
                row['smoker'],
                row['day'],
                row['time'],
                row['size']
            )
            predictions.append(pred)
        
        return np.array(predictions)
    
    def interactive_prediction(self):
        """Interactive prediction interface"""
        print("\n" + "="*60)
        print("WAITER'S TIP PREDICTION - INTERACTIVE MODE")
        print("="*60)
        
        if self.model is None:
            print("\nNo model loaded. Loading default model (Random Forest)...")
            self.load_model('random_forest')
        
        print(f"\nUsing model: {self.model_name}")
        print("\nEnter the following details:")
        
        try:
            total_bill = float(input("Total Bill ($): "))
            sex = input("Gender (Male/Female): ").capitalize()
            smoker = input("Smoker (Yes/No): ").capitalize()
            day = input("Day (Thur/Fri/Sat/Sun): ").capitalize()
            time = input("Time (Lunch/Dinner): ").capitalize()
            size = int(input("Party Size: "))
            
            # Make prediction
            predicted_tip = self.predict_tip(total_bill, sex, smoker, day, time, size)
            
            # Display results
            print("\n" + "="*60)
            print("PREDICTION RESULT")
            print("="*60)
            print(f"\nInput Details:")
            print(f"  Total Bill: ${total_bill:.2f}")
            print(f"  Gender: {sex}")
            print(f"  Smoker: {smoker}")
            print(f"  Day: {day}")
            print(f"  Time: {time}")
            print(f"  Party Size: {size}")
            print(f"\n{'='*60}")
            print(f"Predicted Tip: ${predicted_tip:.2f}")
            print(f"Tip Percentage: {(predicted_tip/total_bill)*100:.1f}%")
            print(f"Total Amount: ${total_bill + predicted_tip:.2f}")
            print("="*60)
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
    
    def demo_predictions(self):
        """Run demo predictions with sample data"""
        print("\n" + "="*60)
        print("DEMO PREDICTIONS")
        print("="*60)
        
        if self.model is None:
            print("\nLoading Random Forest model...")
            self.load_model('random_forest')
        
        # Sample test cases
        test_cases = [
            {'total_bill': 25.50, 'sex': 'Male', 'smoker': 'No', 'day': 'Sat', 'time': 'Dinner', 'size': 2},
            {'total_bill': 48.27, 'sex': 'Female', 'smoker': 'Yes', 'day': 'Fri', 'time': 'Dinner', 'size': 4},
            {'total_bill': 15.04, 'sex': 'Male', 'smoker': 'No', 'day': 'Sun', 'time': 'Lunch', 'size': 3},
            {'total_bill': 35.83, 'sex': 'Female', 'smoker': 'No', 'day': 'Sat', 'time': 'Dinner', 'size': 2},
            {'total_bill': 10.34, 'sex': 'Male', 'smoker': 'Yes', 'day': 'Thur', 'time': 'Lunch', 'size': 1},
        ]
        
        print(f"\nModel: {self.model_name}\n")
        
        for i, case in enumerate(test_cases, 1):
            predicted_tip = self.predict_tip(**case)
            tip_pct = (predicted_tip / case['total_bill']) * 100
            
            print(f"Test Case {i}:")
            print(f"  Bill: ${case['total_bill']:.2f} | {case['sex']} | {case['smoker']} | "
                  f"{case['day']} | {case['time']} | Party: {case['size']}")
            print(f"  → Predicted Tip: ${predicted_tip:.2f} ({tip_pct:.1f}%)")
            print()

def main():
    """Main function for prediction"""
    predictor = TipPredictionSystem()
    
    # Run demo
    predictor.demo_predictions()
    
    # Interactive mode
    while True:
        print("\n" + "="*60)
        choice = input("\nWould you like to make a prediction? (yes/no): ").lower()
        if choice in ['yes', 'y']:
            predictor.interactive_prediction()
        else:
            print("\nThank you for using the Tip Prediction System!")
            break

if __name__ == "__main__":
    main()
