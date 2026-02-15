"""Tests for data schema validation."""

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from src.schema import (
    TrainDataSchema,
    TestDataSchema,
    EngineeredFeaturesSchema,
    SubmissionSchema,
    validate_train_data,
    validate_test_data,
    validate_engineered_features,
    validate_submission,
    get_schema_summary,
)


class TestTrainDataSchema:
    """Tests for training data schema."""

    def test_valid_train_data(self, sample_train_data):
        """Test that valid training data passes schema validation."""
        # Should not raise
        validated = TrainDataSchema.validate(sample_train_data)
        assert len(validated) == len(sample_train_data)

    def test_invalid_age_range(self, sample_train_data):
        """Test that age outside valid range fails validation."""
        df = sample_train_data.copy()
        df.loc[0, "Age"] = 150  # Invalid: > 120

        with pytest.raises(pa.errors.SchemaError):
            TrainDataSchema.validate(df)

    def test_invalid_sex_value(self, sample_train_data):
        """Test that invalid sex value fails validation."""
        df = sample_train_data.copy()
        df.loc[0, "Sex"] = 2  # Invalid: should be 0 or 1

        with pytest.raises(pa.errors.SchemaError):
            TrainDataSchema.validate(df)

    def test_invalid_target_value(self, sample_train_data):
        """Test that invalid target value fails validation."""
        df = sample_train_data.copy()
        df.loc[0, "Heart Disease"] = "Maybe"  # Invalid

        with pytest.raises(pa.errors.SchemaError):
            TrainDataSchema.validate(df)

    def test_missing_values_allowed(self, sample_train_data):
        """Test that missing values are allowed in nullable columns."""
        df = sample_train_data.copy()
        df.loc[0, "Age"] = np.nan
        df.loc[1, "BP"] = np.nan

        # Should not raise
        validated = TrainDataSchema.validate(df)
        assert len(validated) == len(df)

    def test_negative_blood_pressure_fails(self, sample_train_data):
        """Test that negative BP fails validation."""
        df = sample_train_data.copy()
        df.loc[0, "BP"] = -10

        with pytest.raises(pa.errors.SchemaError):
            TrainDataSchema.validate(df)

    def test_cholesterol_out_of_range(self, sample_train_data):
        """Test that extreme cholesterol value fails."""
        df = sample_train_data.copy()
        df.loc[0, "Cholesterol"] = 1000  # > 600

        with pytest.raises(pa.errors.SchemaError):
            TrainDataSchema.validate(df)

    def test_invalid_chest_pain_type(self, sample_train_data):
        """Test that invalid chest pain type fails."""
        df = sample_train_data.copy()
        df.loc[0, "Chest pain type"] = 5  # Should be 1-4

        with pytest.raises(pa.errors.SchemaError):
            TrainDataSchema.validate(df)

    def test_invalid_thallium_value(self, sample_train_data):
        """Test that invalid Thallium value fails."""
        df = sample_train_data.copy()
        df.loc[0, "Thallium"] = 5  # Should be 3, 6, or 7

        with pytest.raises(pa.errors.SchemaError):
            TrainDataSchema.validate(df)


class TestTestDataSchema:
    """Tests for test data schema (no target column)."""

    def test_valid_test_data(self, sample_test_data):
        """Test that valid test data passes validation."""
        validated = TestDataSchema.validate(sample_test_data)
        assert len(validated) == len(sample_test_data)

    def test_test_data_has_no_target(self, sample_test_data):
        """Test that test data schema doesn't require target."""
        assert "Heart Disease" not in sample_test_data.columns
        # Should not raise
        TestDataSchema.validate(sample_test_data)


class TestEngineeredFeaturesSchema:
    """Tests for engineered features schema."""

    def test_valid_engineered_features(self):
        """Test that valid engineered features pass validation."""
        df = pd.DataFrame({
            "Rate_Pressure_Product": [15000.0, 20000.0],
            "Electrical_Stress": [1.5, 2.0],
            "Metabolic_Score": [0, 2],
            "MaxHR_Rel_Age": [0.8, 0.9],
            "MaxHR_x_Age": [10000.0, 12000.0],
            "BP_x_Cholesterol": [30000.0, 40000.0],
            "Age_Bin": [5, 6],
            "Cholesterol_per_Age": [4.5, 3.2],
            "BP_per_Age": [2.5, 2.0],
            "HR_Deficit": [30.0, 50.0],
            "Exercise_Risk": [1.5, 5.0],
            "Vessel_Thallium": [0.0, 14.0],
            "Angina_ST": [0.0, 3.0],
            "Cardiac_Risk": [2, 4],
            "ST_per_HR": [0.01, 0.02],
            "Typical_Angina": [1, 0],
            "Has_Vessel": [0, 1],
            "Thallium_Abnormal": [0, 1],
        })

        validated = EngineeredFeaturesSchema.validate(df)
        assert len(validated) == 2

    def test_metabolic_score_out_of_range(self):
        """Test that Metabolic Score > 3 fails."""
        df = pd.DataFrame({
            "Rate_Pressure_Product": [15000.0],
            "Electrical_Stress": [1.5],
            "Metabolic_Score": [5],  # Invalid: > 3
            "MaxHR_Rel_Age": [0.8],
            "MaxHR_x_Age": [10000.0],
            "BP_x_Cholesterol": [30000.0],
            "Age_Bin": [5],
        })

        with pytest.raises(pa.errors.SchemaError):
            EngineeredFeaturesSchema.validate(df)

    def test_negative_rate_pressure_product(self):
        """Test that negative RPP fails."""
        df = pd.DataFrame({
            "Rate_Pressure_Product": [-100.0],  # Invalid: < 0
            "Electrical_Stress": [1.5],
            "Metabolic_Score": [1],
            "MaxHR_Rel_Age": [0.8],
            "MaxHR_x_Age": [10000.0],
            "BP_x_Cholesterol": [30000.0],
            "Age_Bin": [5],
        })

        with pytest.raises(pa.errors.SchemaError):
            EngineeredFeaturesSchema.validate(df)


class TestSubmissionSchema:
    """Tests for submission schema."""

    def test_valid_submission(self):
        """Test that valid submission data passes validation."""
        df = pd.DataFrame({
            "id": [0, 1, 2],
            "Heart Disease": [0.1, 0.5, 0.95],
        })
        validated = SubmissionSchema.validate(df)
        assert len(validated) == 3

    def test_probability_over_1_fails(self):
        """Test that probability > 1 fails validation."""
        df = pd.DataFrame({
            "id": [0],
            "Heart Disease": [1.5],
        })
        with pytest.raises(pa.errors.SchemaError):
            SubmissionSchema.validate(df)

    def test_validate_submission_function(self):
        """Test validate_submission helper returns correct tuple."""
        df = pd.DataFrame({
            "id": [0, 1],
            "Heart Disease": [0.3, 0.7],
        })
        is_valid, errors = validate_submission(df, raise_error=False)
        assert is_valid is True
        assert errors == []


class TestValidationFunctions:
    """Tests for validation helper functions."""

    def test_validate_train_data_returns_tuple(self, sample_train_data):
        """Test validate_train_data returns correct tuple."""
        is_valid, errors = validate_train_data(sample_train_data, raise_error=False)
        assert is_valid is True
        assert errors == []

    def test_validate_train_data_catches_errors(self, sample_train_data):
        """Test validate_train_data catches errors without raising."""
        df = sample_train_data.copy()
        df.loc[0, "Age"] = 200  # Invalid

        is_valid, errors = validate_train_data(df, raise_error=False)
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_test_data_returns_tuple(self, sample_test_data):
        """Test validate_test_data returns correct tuple."""
        is_valid, errors = validate_test_data(sample_test_data, raise_error=False)
        assert is_valid is True
        assert errors == []

    def test_get_schema_summary_structure(self):
        """Test that schema summary has expected structure."""
        summary = get_schema_summary()

        assert "train" in summary
        assert "test" in summary
        assert "engineered" in summary
        assert "submission" in summary

        # Check train schema has expected columns
        assert "Age" in summary["train"]
        assert "Heart Disease" in summary["train"]

        # Check each column has expected keys
        for col_info in summary["train"].values():
            assert "dtype" in col_info
            assert "nullable" in col_info
            assert "description" in col_info


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test that empty DataFrame with correct columns passes."""
        df = pd.DataFrame(columns=[
            "id", "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
            "FBS over 120", "EKG results", "Max HR", "Exercise angina",
            "ST depression", "Slope of ST", "Number of vessels fluro",
            "Thallium", "Heart Disease"
        ])
        # Empty DF should pass (no rows to validate)
        validated = TrainDataSchema.validate(df)
        assert len(validated) == 0

    def test_boundary_values(self, sample_train_data):
        """Test boundary values pass validation."""
        df = sample_train_data.copy()
        # Set boundary values
        df.loc[0, "Age"] = 0      # min
        df.loc[1, "Age"] = 120    # max
        df.loc[0, "BP"] = 0       # min
        df.loc[0, "Cholesterol"] = 600  # max

        validated = TrainDataSchema.validate(df)
        assert len(validated) == len(df)

    def test_all_nulls_in_nullable_column(self, sample_train_data):
        """Test column with all nulls passes if nullable."""
        df = sample_train_data.copy()
        df["Age"] = np.nan

        validated = TrainDataSchema.validate(df)
        assert len(validated) == len(df)

    def test_type_coercion(self, sample_train_data):
        """Test that type coercion works correctly."""
        df = sample_train_data.copy()
        # Pass integers as strings (should be coerced)
        df["Age"] = df["Age"].astype(str)

        validated = TrainDataSchema.validate(df)
        assert validated["Age"].dtype == float
