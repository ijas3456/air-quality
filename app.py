from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or use your specific React origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model
model = joblib.load("air_quality_model.pkl")

# Input schema using Pydantic
class InputData(BaseModel):
    CO: float
    NO2: float
    T: float
    RH: float

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.CO, data.NO2, data.T, data.RH]])
    prediction = model.predict(features)[0]
    return {"predicted_benzene": round(prediction, 2)}
