from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Kaggle Playground S6E2 - CatBoost Model",
    version="1.0.0"
)

# モデルのロード
MODEL_PATH = "output/models/model_cat.pkl"

class PredictionInput(BaseModel):
    Age: float
    BP: float
    Cholesterol: float
    Max_HR: float
    ST_depression: float
    Number_of_vessels_fluro: float
    Sex: int
    Chest_pain_type: int
    FBS_over_120: int
    EKG_results: int
    Exercise_angina: int
    Slope_of_ST: int
    Thallium: int

class PredictionOutput(BaseModel):
    probability: float
    prediction: int

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Model not found")

    pipeline = joblib.load(MODEL_PATH)

    df = pd.DataFrame([input_data.model_dump()])
    # カラム名を元に戻す
    df.columns = [c.replace('_', ' ') for c in df.columns]

    prob = pipeline.predict_proba(df)[0, 1]
    pred = int(prob >= 0.5)

    return PredictionOutput(probability=prob, prediction=pred)
