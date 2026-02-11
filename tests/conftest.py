"""Pytest fixtures for testing."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_train_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "id": range(n_samples),
        "Age": np.random.randint(30, 80, n_samples),
        "Sex": np.random.choice([0, 1], n_samples),
        "Chest pain type": np.random.choice([1, 2, 3, 4], n_samples),
        "BP": np.random.randint(90, 180, n_samples),
        "Cholesterol": np.random.randint(100, 400, n_samples),
        "FBS over 120": np.random.choice([0, 1], n_samples),
        "EKG results": np.random.choice([0, 1, 2], n_samples),
        "Max HR": np.random.randint(80, 200, n_samples),
        "Exercise angina": np.random.choice([0, 1], n_samples),
        "ST depression": np.random.uniform(0, 5, n_samples),
        "Slope of ST": np.random.choice([1, 2, 3], n_samples),
        "Number of vessels fluro": np.random.choice([0, 1, 2, 3], n_samples),
        "Thallium": np.random.choice([3, 6, 7], n_samples),
        "Heart Disease": np.random.choice(["Presence", "Absence"], n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_test_data(sample_train_data):
    """Create sample test data (without target)."""
    df = sample_train_data.copy()
    return df.drop(columns=["Heart Disease"])
