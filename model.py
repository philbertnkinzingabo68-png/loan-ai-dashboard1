import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import os
from pathlib import Path

class LoanPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.model_path = str(Path(__file__).parent / 'model.pkl')
        self.scaler_path = str(Path(__file__).parent / 'scaler.pkl')
        self.encoders_path = str(Path(__file__).parent / 'label_encoders.pkl')
        self._load_model()
        self._load_scaler()
        self._load_encoders()
    
    def _load_model(self):
        """Load the model from disk"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print("Model loaded successfully")
            else:
                print(f"Model file not found at {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None

    def _load_scaler(self):
        """Load the scaler from disk"""
        try:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print("Scaler loaded successfully")
            else:
                print(f"Scaler file not found at {self.scaler_path}")
        except Exception as e:
            print(f"Error loading scaler: {str(e)}")
            self.scaler = None
    
    def _load_encoders(self):
        """Load label encoders from disk"""
        try:
            if os.path.exists(self.encoders_path):
                self.label_encoders = joblib.load(self.encoders_path)
                print("Label encoders loaded successfully")
                
                # Print what values each encoder expects
                print("\nExpected values for categorical features:")
                for key, encoder in self.label_encoders.items():
                    if hasattr(encoder, 'classes_'):
                        print(f"  {key}: {list(encoder.classes_)}")
            else:
                print(f"Label encoders file not found at {self.encoders_path}")
                # Create default encoders based on your data
                self.label_encoders = {
                    'gender': LabelEncoder().fit(['Male', 'Female']),
                    'marital_status': LabelEncoder().fit(['Single', 'Married', 'Divorced', 'Widowed']),
                    'education_level': LabelEncoder().fit(['High School', "Bachelor's", "Master's", 'PhD']),
                    'employment_status': LabelEncoder().fit(['Employed', 'Self-employed', 'Unemployed']),
                    'loan_purpose': LabelEncoder().fit(['Debt consolidation', 'Other', 'Car', 'Home', 'Education', 'Business', 'Medical', 'Vacation']),
                    'grade_subgrade': LabelEncoder().fit(['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'F1'])
                }
        except Exception as e:
            print(f"Error loading encoders: {str(e)}")
            self.label_encoders = {}
    
    def _normalize_categorical_value(self, column, value):
        """Normalize categorical values to match encoder expectations"""
        if not value or not isinstance(value, str):
            return value
        
        # Get expected values from encoder
        if column in self.label_encoders and hasattr(self.label_encoders[column], 'classes_'):
            expected_values = list(self.label_encoders[column].classes_)
            
            # Try exact match first
            if value in expected_values:
                return value
            
            # Try case-insensitive match
            value_lower = value.lower().strip()
            for expected in expected_values:
                if expected.lower() == value_lower:
                    print(f"Normalized '{value}' to '{expected}' for {column}")
                    return expected
            
            # Special handling for common variations
            if column == 'loan_purpose':
                # Handle common variations in loan purpose
                variations = {
                    'home improvement': 'Home',
                    'home_improvement': 'Home',
                    'homeimprovement': 'Home',
                    'debt_consolidation': 'Debt consolidation',
                    'debtconsolidation': 'Debt consolidation',
                }
                
                if value_lower in variations:
                    result = variations[value_lower]
                    print(f"Normalized '{value}' to '{result}' for {column}")
                    return result
                
                # Try to find match in expected values
                for expected in expected_values:
                    if expected.lower().replace(' ', '').replace('_', '') == value_lower.replace(' ', '').replace('_', ''):
                        print(f"Normalized '{value}' to '{expected}' for {column}")
                        return expected
        
        return value
    
    def predict_single(self, features):
        """
        Make prediction for a single loan application.
        Uses ONLY the trained machine learning model - NO business rules.
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not loaded. Please check if model.pkl and scaler.pkl exist.")
            
        try:
            # Feature order matching your trained model
            feature_order = [
                'annual_income', 'debt_to_income_ratio', 'credit_score', 
                'loan_amount', 'interest_rate', 'gender', 'marital_status', 
                'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade'
            ]
            
            # Categorical columns that need encoding
            categorical_cols = ['gender', 'marital_status', 'education_level', 
                              'employment_status', 'loan_purpose', 'grade_subgrade']
            
            # Encode categorical features
            encoded_features = features.copy()
            for col in categorical_cols:
                if col in encoded_features and col in self.label_encoders:
                    try:
                        # Normalize the value before encoding
                        normalized_value = self._normalize_categorical_value(col, encoded_features[col])
                        encoded_features[col] = self.label_encoders[col].transform([normalized_value])[0]
                    except ValueError as e:
                        # Get expected values for better error message
                        expected = list(self.label_encoders[col].classes_) if hasattr(self.label_encoders[col], 'classes_') else 'unknown'
                        raise ValueError(f"Invalid value for {col}: '{encoded_features[col]}'. Expected one of: {expected}")
            
            # Create DataFrame with feature names
            features_df = pd.DataFrame([[encoded_features[col] for col in feature_order]], 
                                      columns=feature_order)
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Get ML model prediction - PURE ML, NO RULES
            prediction = self.model.predict(features_scaled)[0]
            proba = self.model.predict_proba(features_scaled)[0]
            
            # Get probability for the predicted class
            if len(proba) == 1:
                probability = proba[0]
            else:
                probability = proba[1]  # Probability of class 1 (approved)
            
            # Log the decision
            print(f"\n--- Pure ML Prediction ---")
            print(f"Prediction: {'APPROVED' if prediction == 1 else 'REJECTED'}")
            print(f"Confidence: {probability:.2%}")
            print(f"-------------------------\n")
            
            return int(prediction), float(probability)
            
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")