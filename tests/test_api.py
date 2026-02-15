"""Tests for FastAPI endpoints."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, _model_cache, _classify_risk


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    """TestClient with no model loaded."""
    _model_cache["pipeline"] = None
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    _model_cache["pipeline"] = None


@pytest.fixture
def mock_model():
    """Mock pipeline that returns controllable probabilities."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture
def client_with_model(mock_model):
    """TestClient with a mock model loaded."""
    _model_cache["pipeline"] = mock_model
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    _model_cache["pipeline"] = None


VALID_INPUT = {
    "Age": 55,
    "Sex": 1,
    "Chest pain type": 4,
    "BP": 130,
    "Cholesterol": 250,
    "FBS over 120": 0,
    "EKG results": 0,
    "Max HR": 150,
    "Exercise angina": 0,
    "ST depression": 1.5,
    "Slope of ST": 2,
    "Number of vessels fluro": 1,
    "Thallium": 3,
}


# =============================================================================
# /health tests
# =============================================================================

class TestHealth:
    def test_health_no_model(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False
        assert data["version"] == "2.0.0"

    def test_health_with_model(self, client_with_model):
        resp = client_with_model.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is True


# =============================================================================
# /predict tests
# =============================================================================

class TestPredict:
    def test_predict_success(self, client_with_model, mock_model):
        resp = client_with_model.post("/predict", json=VALID_INPUT)
        assert resp.status_code == 200
        data = resp.json()
        assert data["probability"] == pytest.approx(0.7, abs=1e-4)
        assert data["prediction"] == 1
        assert data["risk_level"] == "High"
        mock_model.predict_proba.assert_called_once()

    def test_predict_no_model(self, client):
        resp = client.post("/predict", json=VALID_INPUT)
        assert resp.status_code == 503
        assert "Model not loaded" in resp.json()["detail"]

    def test_predict_age_over_120(self, client_with_model):
        bad = {**VALID_INPUT, "Age": 150}
        resp = client_with_model.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_invalid_sex(self, client_with_model):
        bad = {**VALID_INPUT, "Sex": 2}
        resp = client_with_model.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_missing_field(self, client_with_model):
        incomplete = {k: v for k, v in VALID_INPUT.items() if k != "Age"}
        resp = client_with_model.post("/predict", json=incomplete)
        assert resp.status_code == 422

    def test_predict_negative_bp(self, client_with_model):
        bad = {**VALID_INPUT, "BP": -10}
        resp = client_with_model.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_cholesterol_too_high(self, client_with_model):
        bad = {**VALID_INPUT, "Cholesterol": 700}
        resp = client_with_model.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_low_probability(self, client_with_model, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        resp = client_with_model.post("/predict", json=VALID_INPUT)
        data = resp.json()
        assert data["prediction"] == 0
        assert data["risk_level"] == "Low"


# =============================================================================
# /predict/batch tests
# =============================================================================

class TestBatchPredict:
    def test_batch_predict_success(self, client_with_model, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        resp = client_with_model.post("/predict/batch", json=[VALID_INPUT, VALID_INPUT])
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
        assert data["processing_time_ms"] >= 0

    def test_batch_predict_no_model(self, client):
        resp = client.post("/predict/batch", json=[VALID_INPUT])
        assert resp.status_code == 503

    def test_batch_predict_validation_error(self, client_with_model):
        bad_input = {**VALID_INPUT, "Age": 999}
        resp = client_with_model.post("/predict/batch", json=[bad_input])
        assert resp.status_code == 422


# =============================================================================
# /model/info tests
# =============================================================================

class TestModelInfo:
    def test_model_info(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_type"] == "CatBoost"
        assert len(data["features"]) == 13
        assert data["best_cv_auc"] > 0.95
        assert data["best_lb_score"] > 0.95

    def test_model_info_features_complete(self, client):
        resp = client.get("/model/info")
        features = resp.json()["features"]
        assert "Age" in features
        assert "Thallium" in features
        assert "Max HR" in features


# =============================================================================
# Helper tests
# =============================================================================

class TestHelpers:
    def test_classify_risk_low(self):
        assert _classify_risk(0.1) == "Low"
        assert _classify_risk(0.29) == "Low"

    def test_classify_risk_medium(self):
        assert _classify_risk(0.3) == "Medium"
        assert _classify_risk(0.5) == "Medium"
        assert _classify_risk(0.69) == "Medium"

    def test_classify_risk_high(self):
        assert _classify_risk(0.7) == "High"
        assert _classify_risk(0.99) == "High"
