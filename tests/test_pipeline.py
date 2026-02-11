"""Tests for model pipeline functions."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import get_pipeline, get_feature_names, create_engineered_features


class TestGetPipeline:
    """Test pipeline creation."""

    @pytest.mark.parametrize("model_name", ["rf", "lgbm", "xgb"])
    def test_pipeline_creation(self, model_name):
        """Test pipeline is created for different models."""
        result = get_pipeline(model_name, use_engineered=False)
        pipeline = result[0] if isinstance(result, tuple) else result
        assert isinstance(pipeline, Pipeline)

    def test_catboost_returns_tuple(self):
        """Test CatBoost returns preprocessor and model separately."""
        result = get_pipeline("cat", use_engineered=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        preprocessor, model = result
        assert model is not None

    def test_invalid_model_raises_error(self):
        """Test invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_pipeline("invalid_model")

    @pytest.mark.parametrize("model_name", ["rf", "lgbm", "xgb"])
    def test_pipeline_can_fit(self, model_name, sample_train_data):
        """Test pipeline can fit on sample data."""
        numerical, categorical = get_feature_names(use_engineered=False)
        X = sample_train_data[numerical + categorical]
        y = sample_train_data["Heart Disease"].map({"Presence": 1, "Absence": 0})

        result = get_pipeline(model_name, use_engineered=False)
        pipeline = result[0] if isinstance(result, tuple) else result

        # Should not raise
        pipeline.fit(X, y)

        # Should be able to predict
        preds = pipeline.predict_proba(X)
        assert preds.shape == (len(X), 2)
        assert np.all((preds >= 0) & (preds <= 1))


class TestPipelineWithEngineeredFeatures:
    """Test pipeline with engineered features."""

    def test_pipeline_with_engineered_features(self, sample_train_data):
        """Test pipeline works with engineered features."""
        df = create_engineered_features(sample_train_data)
        numerical, categorical = get_feature_names(use_engineered=True)
        X = df[numerical + categorical]
        y = df["Heart Disease"].map({"Presence": 1, "Absence": 0})

        result = get_pipeline("rf", use_engineered=True)
        pipeline = result[0]

        pipeline.fit(X, y)
        preds = pipeline.predict_proba(X)

        assert preds.shape == (len(X), 2)
