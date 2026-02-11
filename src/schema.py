"""Data schema definitions using Pandera.

This module defines the expected schema for input data, ensuring data quality
and catching issues early in the pipeline.
"""

import pandera as pa
from pandera import Column, Check, DataFrameSchema
from pandera.typing import Series
import pandas as pd


# =============================================================================
# Schema Definitions
# =============================================================================

TrainDataSchema = DataFrameSchema(
    columns={
        # ID column
        "id": Column(
            int,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Unique identifier for each record",
        ),
        # Numerical features
        "Age": Column(
            float,
            Check.in_range(0, 120),
            nullable=True,
            coerce=True,
            description="Patient age in years",
        ),
        "BP": Column(
            float,
            Check.in_range(0, 300),
            nullable=True,
            coerce=True,
            description="Blood pressure (systolic)",
        ),
        "Cholesterol": Column(
            float,
            Check.in_range(0, 600),
            nullable=True,
            coerce=True,
            description="Serum cholesterol in mg/dl",
        ),
        "Max HR": Column(
            float,
            Check.in_range(0, 250),
            nullable=True,
            coerce=True,
            description="Maximum heart rate achieved",
        ),
        "ST depression": Column(
            float,
            Check.in_range(0, 10),
            nullable=True,
            coerce=True,
            description="ST depression induced by exercise relative to rest",
        ),
        "Number of vessels fluro": Column(
            float,
            Check.in_range(0, 4),
            nullable=True,
            coerce=True,
            description="Number of major vessels colored by fluoroscopy (0-3)",
        ),
        # Categorical features
        "Sex": Column(
            float,
            Check.isin([0, 1]),
            nullable=True,
            coerce=True,
            description="Sex (0=female, 1=male)",
        ),
        "Chest pain type": Column(
            float,
            Check.isin([1, 2, 3, 4]),
            nullable=True,
            coerce=True,
            description="Chest pain type (1-4)",
        ),
        "FBS over 120": Column(
            float,
            Check.isin([0, 1]),
            nullable=True,
            coerce=True,
            description="Fasting blood sugar > 120 mg/dl",
        ),
        "EKG results": Column(
            float,
            Check.isin([0, 1, 2]),
            nullable=True,
            coerce=True,
            description="Resting electrocardiographic results (0-2)",
        ),
        "Exercise angina": Column(
            float,
            Check.isin([0, 1]),
            nullable=True,
            coerce=True,
            description="Exercise induced angina",
        ),
        "Slope of ST": Column(
            float,
            Check.isin([1, 2, 3]),
            nullable=True,
            coerce=True,
            description="Slope of peak exercise ST segment",
        ),
        "Thallium": Column(
            float,
            Check.isin([3, 6, 7]),
            nullable=True,
            coerce=True,
            description="Thallium stress test result",
        ),
        # Target
        "Heart Disease": Column(
            str,
            Check.isin(["Presence", "Absence"]),
            nullable=False,
            description="Target variable: presence of heart disease",
        ),
    },
    strict=False,  # Allow extra columns
    coerce=True,
)


TestDataSchema = TrainDataSchema.remove_columns(["Heart Disease"])


# =============================================================================
# Schema for Engineered Features
# =============================================================================

EngineeredFeaturesSchema = DataFrameSchema(
    columns={
        "Rate_Pressure_Product": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=True,
            coerce=True,
            description="BP × Max HR (myocardial oxygen demand)",
        ),
        "Electrical_Stress": Column(
            float,
            nullable=True,
            coerce=True,
            description="ST depression × Slope of ST",
        ),
        "Metabolic_Score": Column(
            int,
            Check.in_range(0, 3),
            nullable=True,
            coerce=True,
            description="Composite metabolic risk score (0-3)",
        ),
        "MaxHR_Rel_Age": Column(
            float,
            Check.in_range(0, 2),
            nullable=True,
            coerce=True,
            description="Max HR / (220 - Age)",
        ),
        "MaxHR_x_Age": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=True,
            coerce=True,
            description="Max HR × Age interaction",
        ),
        "BP_x_Cholesterol": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=True,
            coerce=True,
            description="BP × Cholesterol interaction",
        ),
        "Age_Bin": Column(
            int,
            Check.in_range(0, 12),
            nullable=True,
            coerce=True,
            description="Age binned by decade",
        ),
    },
    strict=False,
    coerce=True,
)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_train_data(df: pd.DataFrame, raise_error: bool = True) -> tuple[bool, list[str]]:
    """Validate training data against schema.

    Args:
        df: DataFrame to validate
        raise_error: If True, raise SchemaError on failure

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    try:
        TrainDataSchema.validate(df, lazy=True)
        return True, []
    except pa.errors.SchemaErrors as e:
        errors = [str(err) for err in e.failure_cases.to_dict("records")]
        if raise_error:
            raise
        return False, errors


def validate_test_data(df: pd.DataFrame, raise_error: bool = True) -> tuple[bool, list[str]]:
    """Validate test data against schema.

    Args:
        df: DataFrame to validate
        raise_error: If True, raise SchemaError on failure

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    try:
        TestDataSchema.validate(df, lazy=True)
        return True, []
    except pa.errors.SchemaErrors as e:
        errors = [str(err) for err in e.failure_cases.to_dict("records")]
        if raise_error:
            raise
        return False, errors


def validate_engineered_features(df: pd.DataFrame, raise_error: bool = True) -> tuple[bool, list[str]]:
    """Validate engineered features against schema.

    Args:
        df: DataFrame to validate
        raise_error: If True, raise SchemaError on failure

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    try:
        EngineeredFeaturesSchema.validate(df, lazy=True)
        return True, []
    except pa.errors.SchemaErrors as e:
        errors = [str(err) for err in e.failure_cases.to_dict("records")]
        if raise_error:
            raise
        return False, errors


def get_schema_summary() -> dict:
    """Get a summary of all schema definitions.

    Returns:
        Dictionary with schema information
    """
    def schema_to_dict(schema: DataFrameSchema) -> dict:
        return {
            col_name: {
                "dtype": str(col.dtype),
                "nullable": col.nullable,
                "description": col.description or "",
            }
            for col_name, col in schema.columns.items()
        }

    return {
        "train": schema_to_dict(TrainDataSchema),
        "test": schema_to_dict(TestDataSchema),
        "engineered": schema_to_dict(EngineeredFeaturesSchema),
    }


if __name__ == "__main__":
    # Print schema summary
    import json
    print(json.dumps(get_schema_summary(), indent=2))
