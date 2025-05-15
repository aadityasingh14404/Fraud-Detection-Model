from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Credit Card Fraud Detection API")

# Load the trained model and preprocessors
model = joblib.load('random_forest_model.pkl')
power_transformer = joblib.load('power_transformer.pkl')
robust_scaler = joblib.load('robust_scaler.pkl')

# Define input data model
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Credit Card Fraud Detection API"}

# Prediction endpoint
@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([transaction.dict()])
        
        # Identify skewed columns (based on training)
        skewed_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                         'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V20', 
                         'V21', 'V23', 'V27', 'V28', 'Amount']
        
        # Apply PowerTransformer to skewed columns
        if skewed_columns:
            data[skewed_columns] = power_transformer.transform(data[skewed_columns])
        
        # Apply RobustScaler to Amount
        data[['Amount']] = robust_scaler.transform(data[['Amount']])
        
        # Make prediction
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]
        
        # Return result
        result = "Fraudulent" if prediction == 1 else "Non-Fraudulent"
        return {
            "prediction": result,
            "fraud_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))