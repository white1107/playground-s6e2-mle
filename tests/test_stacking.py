"""Tests for stacking ensemble module."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from src.stacking import build_meta_features, _get_meta_learner, _predict_proba


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_preds():
    """Sample base-model predictions for 3 models, 100 samples."""
    np.random.seed(42)
    return {
        "cat": np.random.uniform(0, 1, 100),
        "lgbm": np.random.uniform(0, 1, 100),
        "xgb": np.random.uniform(0, 1, 100),
    }


@pytest.fixture
def binary_labels():
    np.random.seed(42)
    return np.random.choice([0, 1], 100)


# =============================================================================
# build_meta_features
# =============================================================================

class TestBuildMetaFeatures:
    def test_output_shape(self, sample_preds):
        meta = build_meta_features(sample_preds)
        n = 100
        m = 3  # number of models
        # raw(3) + mean(1) + std(1) + spread(1) + ranks(3) = 9
        assert meta.shape == (n, m + 3 + m)

    def test_mean_column(self, sample_preds):
        meta = build_meta_features(sample_preds)
        m = 3
        # mean is at column index m (after raw predictions)
        mean_col = meta[:, m]
        names = sorted(sample_preds.keys())
        expected = np.column_stack([sample_preds[n] for n in names]).mean(axis=1)
        np.testing.assert_allclose(mean_col, expected)

    def test_std_column(self, sample_preds):
        meta = build_meta_features(sample_preds)
        m = 3
        std_col = meta[:, m + 1]
        names = sorted(sample_preds.keys())
        expected = np.column_stack([sample_preds[n] for n in names]).std(axis=1)
        np.testing.assert_allclose(std_col, expected)

    def test_spread_column(self, sample_preds):
        meta = build_meta_features(sample_preds)
        m = 3
        spread_col = meta[:, m + 2]
        names = sorted(sample_preds.keys())
        raw = np.column_stack([sample_preds[n] for n in names])
        expected = raw.max(axis=1) - raw.min(axis=1)
        np.testing.assert_allclose(spread_col, expected)

    def test_rank_columns_range(self, sample_preds):
        meta = build_meta_features(sample_preds)
        m = 3
        # ranks are at columns m+3 .. m+3+m-1
        rank_cols = meta[:, m + 3:]
        assert rank_cols.shape[1] == m
        # ranks should be in (0, 1] (percentile)
        assert rank_cols.min() > 0
        assert rank_cols.max() <= 1.0

    def test_two_models(self):
        preds = {
            "a": np.array([0.1, 0.9, 0.5]),
            "b": np.array([0.2, 0.8, 0.4]),
        }
        meta = build_meta_features(preds)
        assert meta.shape == (3, 2 + 3 + 2)  # raw(2) + stats(3) + ranks(2)


# =============================================================================
# _get_meta_learner
# =============================================================================

class TestGetMetaLearner:
    def test_lr(self):
        model = _get_meta_learner("lr")
        assert isinstance(model, LogisticRegression)

    def test_ridge(self):
        model = _get_meta_learner("ridge")
        assert isinstance(model, RidgeClassifier)

    def test_lgbm(self):
        model = _get_meta_learner("lgbm")
        assert model.__class__.__name__ == "LGBMClassifier"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown meta-learner"):
            _get_meta_learner("invalid")


# =============================================================================
# _predict_proba
# =============================================================================

class TestPredictProba:
    def test_logistic_regression(self, sample_preds, binary_labels):
        meta = build_meta_features(sample_preds)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(meta, binary_labels)
        probs = _predict_proba(lr, meta)
        assert probs.shape == (100,)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_ridge_classifier(self, sample_preds, binary_labels):
        meta = build_meta_features(sample_preds)
        ridge = RidgeClassifier()
        ridge.fit(meta, binary_labels)
        probs = _predict_proba(ridge, meta)
        assert probs.shape == (100,)
        # Sigmoid output should be in (0, 1)
        assert np.all((probs > 0) & (probs < 1))
