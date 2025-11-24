# train_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib
from sklearn.metrics import brier_score_loss

# Load training data
def train_and_save_model():
    """
    Train multi-factor regression model with PT, APTT, D-Dimer, MPV
    Apply isotonic calibration and save the model
    """
    try:
        # Read training data
        train_data = pd.read_excel('train.xlsx')
        
        # Define features and target
        features = ['PT', 'APTT', 'D-Dimer', 'MPV']
        target = 'DIC'
        
        # Check if required columns exist
        missing_cols = [col for col in features + [target] if col not in train_data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Prepare data
        X_train = train_data[features]
        y_train = train_data[target]
        
        # Handle missing values
        if X_train.isnull().any().any():
            print("Warning: Missing values found. Filling with median.")
            X_train = X_train.fillna(X_train.median())
        
        # Train base logistic regression model
        base_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        # Fit base model
        base_model.fit(X_train, y_train)
        
        # Apply isotonic calibration
        calibrated_model = CalibratedClassifierCV(
            base_model, 
            method='isotonic', 
            cv=5
        )
        calibrated_model.fit(X_train, y_train)
        
        # Save the calibrated model
        model_info = {
            'model': calibrated_model,
            'features': features,
            'risk_thresholds': [0.222, 0.640]
        }
        
        joblib.dump(model_info, 'PADM_model.pkl')
        
        # Evaluate model performance
        y_pred_proba = calibrated_model.predict_proba(X_train)[:, 1]
        brier_score = brier_score_loss(y_train, y_pred_proba)
        
        print("Model training completed successfully!")
        print(f"Brier score: {brier_score:.4f}")
        print(f"Features used: {features}")
        print(f"Risk thresholds: Low < 0.222, Medium 0.222-0.640, High > 0.640")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model()