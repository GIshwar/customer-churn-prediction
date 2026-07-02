import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load model and threshold
model = joblib.load("../models/xgb_final_model.pkl")
threshold = joblib.load("../models/threshold.pkl")

app = FastAPI(title="Customer Churn Prediction API")

# Define input schema - all features the model expects
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    MonthlyCharges: float
    AvgMonthlySpend: float
    InternetService_DSL: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_Month_to_month: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Bank_transfer_automatic: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(customer: CustomerData):
    features = np.array([[
        customer.gender, customer.SeniorCitizen, customer.Partner,
        customer.Dependents, customer.tenure, customer.PhoneService,
        customer.MultipleLines, customer.OnlineSecurity, customer.OnlineBackup,
        customer.DeviceProtection, customer.TechSupport, customer.StreamingTV,
        customer.StreamingMovies, customer.PaperlessBilling, customer.MonthlyCharges,
        customer.AvgMonthlySpend, customer.InternetService_DSL,
        customer.InternetService_Fiber_optic, customer.InternetService_No,
        customer.Contract_Month_to_month, customer.Contract_One_year,
        customer.Contract_Two_year, customer.PaymentMethod_Bank_transfer_automatic,
        customer.PaymentMethod_Credit_card_automatic,
        customer.PaymentMethod_Electronic_check, customer.PaymentMethod_Mailed_check
    ]])

    prob = model.predict_proba(features)[0][1]
    prediction = int(prob >= threshold)

    return {
        "churn_probability": round(float(prob), 4),
        "churn_prediction": prediction,
        "risk_level": "High" if prob >= 0.6 else "Medium" if prob >= 0.4 else "Low"
    }