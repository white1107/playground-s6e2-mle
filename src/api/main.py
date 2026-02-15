"""FastAPI application for Heart Disease Prediction.

Provides endpoints for single/batch prediction, health check, and model info.
"""

import os
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.feature_engineering import add_domain_features

# =============================================================================
# Pydantic models
# =============================================================================

FEATURE_COLUMNS = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
    "FBS over 120", "EKG results", "Max HR", "Exercise angina",
    "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium",
]


class PredictionInput(BaseModel):
    Age: float = Field(..., ge=0, le=120, description="Patient age in years")
    Sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    Chest_pain_type: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)", alias="Chest pain type")
    BP: float = Field(..., ge=0, le=300, description="Blood pressure (systolic)")
    Cholesterol: float = Field(..., ge=0, le=600, description="Serum cholesterol in mg/dl")
    FBS_over_120: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl", alias="FBS over 120")
    EKG_results: int = Field(..., ge=0, le=2, description="Resting EKG results (0-2)", alias="EKG results")
    Max_HR: float = Field(..., ge=0, le=250, description="Maximum heart rate achieved", alias="Max HR")
    Exercise_angina: int = Field(..., ge=0, le=1, description="Exercise induced angina", alias="Exercise angina")
    ST_depression: float = Field(..., ge=0, le=10, description="ST depression", alias="ST depression")
    Slope_of_ST: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment", alias="Slope of ST")
    Number_of_vessels_fluro: float = Field(
        ..., ge=0, le=4, description="Number of major vessels (0-3)", alias="Number of vessels fluro"
    )
    Thallium: int = Field(..., ge=3, le=7, description="Thallium stress test result (3, 6, or 7)")

    model_config = {"populate_by_name": True}


class PredictionOutput(BaseModel):
    probability: float = Field(..., description="Probability of heart disease (0-1)")
    prediction: int = Field(..., description="Binary prediction (0 or 1)")
    risk_level: str = Field(..., description="Risk level: Low / Medium / High")


class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]
    count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfo(BaseModel):
    model_type: str
    features: list[str]
    best_cv_auc: float
    best_lb_score: float


# =============================================================================
# Model cache
# =============================================================================

_model_cache: dict = {"pipeline": None}

MODEL_PATH = os.environ.get("MODEL_PATH", "output/models/model_cat.pkl")


def load_model():
    """Load model from disk into cache."""
    if os.path.exists(MODEL_PATH):
        _model_cache["pipeline"] = joblib.load(MODEL_PATH)


def get_model():
    """Return cached model or None."""
    return _model_cache["pipeline"]


# =============================================================================
# Feature preparation
# =============================================================================

def _prepare_features(input_data: PredictionInput) -> pd.DataFrame:
    """Convert input to DataFrame and apply domain feature engineering."""
    data = input_data.model_dump(by_alias=True)
    df = pd.DataFrame([data])
    df = add_domain_features(df)
    return df


def _classify_risk(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    if probability < 0.7:
        return "Medium"
    return "High"


# =============================================================================
# App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Kaggle Playground S6E2 - CatBoost Model Serving",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Exception handler
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=get_model() is not None,
        version=app.version,
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = _prepare_features(input_data)
    prob = float(model.predict_proba(df)[0, 1])
    pred = int(prob >= 0.5)

    return PredictionOutput(
        probability=round(prob, 6),
        prediction=pred,
        risk_level=_classify_risk(prob),
    )


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(inputs: list[PredictionInput]):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not inputs:
        raise HTTPException(status_code=422, detail="Empty batch")

    start = time.perf_counter()

    rows = [inp.model_dump(by_alias=True) for inp in inputs]
    df = pd.DataFrame(rows)
    df = add_domain_features(df)

    probs = model.predict_proba(df)[:, 1]
    results = []
    for p in probs:
        p_float = float(p)
        results.append(PredictionOutput(
            probability=round(p_float, 6),
            prediction=int(p_float >= 0.5),
            risk_level=_classify_risk(p_float),
        ))

    elapsed_ms = (time.perf_counter() - start) * 1000

    return BatchPredictionOutput(
        predictions=results,
        count=len(results),
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    return ModelInfo(
        model_type="CatBoost",
        features=FEATURE_COLUMNS,
        best_cv_auc=0.95549,
        best_lb_score=0.95395,
    )
