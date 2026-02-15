
import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import add_original_statistics, add_domain_features, get_domain_features


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def sample_df():
    """Minimal sample data that covers all feature engineering paths."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4],
        'Age': [45, 60, 30, 75],
        'Sex': [1, 0, 1, 0],
        'Chest pain type': [4, 2, 1, 4],
        'BP': [150, 120, 100, 160],
        'Cholesterol': [250, 180, 300, 210],
        'FBS over 120': [1, 0, 0, 1],
        'EKG results': [0, 1, 2, 0],
        'Max HR': [170, 140, 190, 100],
        'Exercise angina': [0, 1, 0, 1],
        'ST depression': [1.5, 3.0, 0.0, 2.5],
        'Slope of ST': [2, 3, 1, 3],
        'Number of vessels fluro': [0, 2, 0, 3],
        'Thallium': [3, 7, 3, 6],
        'Heart Disease': [0, 1, 0, 1],
    })


# =========================================================================
# Original statistics tests
# =========================================================================

def test_original_stats():
    original_df = pd.DataFrame({
        'Age': [50, 50, 50, 60, 60],
        'Heart Disease': [1, 1, 0, 0, 0]
    })
    train_df = pd.DataFrame({
        'Age': [50, 60, 70],
        'id': [1, 2, 3]
    })

    base_features = ['Age']
    result_df = add_original_statistics(train_df, original_df, base_features)

    assert 'orig_Age_mean' in result_df.columns

    age50_row = result_df[result_df['Age'] == 50].iloc[0]
    assert np.isclose(age50_row['orig_Age_mean'], 2/3)
    assert age50_row['orig_Age_count'] == 3

    age60_row = result_df[result_df['Age'] == 60].iloc[0]
    assert np.isclose(age60_row['orig_Age_mean'], 0.0)
    assert age60_row['orig_Age_count'] == 2

    age70_row = result_df[result_df['Age'] == 70].iloc[0]
    global_mean = original_df['Heart Disease'].mean()
    assert np.isclose(age70_row['orig_Age_mean'], global_mean)
    assert age70_row['orig_Age_count'] == 0


# =========================================================================
# Domain feature tests
# =========================================================================

class TestDomainFeatures:
    """Tests for add_domain_features."""

    def test_all_features_created(self, sample_df):
        result = add_domain_features(sample_df)
        expected = get_domain_features()
        for feat in expected:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_does_not_modify_input(self, sample_df):
        original_cols = list(sample_df.columns)
        add_domain_features(sample_df)
        assert list(sample_df.columns) == original_cols

    def test_rate_pressure_product(self, sample_df):
        result = add_domain_features(sample_df)
        expected = sample_df['BP'] * sample_df['Max HR']
        pd.testing.assert_series_equal(result['Rate_Pressure_Product'], expected, check_names=False)

    def test_metabolic_score_range(self, sample_df):
        result = add_domain_features(sample_df)
        assert result['Metabolic_Score'].between(0, 3).all()

    def test_maxhr_rel_age(self, sample_df):
        result = add_domain_features(sample_df)
        assert (result['MaxHR_Rel_Age'] > 0).all()

    def test_hr_deficit(self, sample_df):
        result = add_domain_features(sample_df)
        expected = (220 - sample_df['Age']) - sample_df['Max HR']
        pd.testing.assert_series_equal(result['HR_Deficit'], expected, check_names=False)

    def test_cholesterol_per_age(self, sample_df):
        result = add_domain_features(sample_df)
        expected = sample_df['Cholesterol'] / sample_df['Age']
        pd.testing.assert_series_equal(result['Cholesterol_per_Age'], expected, check_names=False)

    def test_vessel_thallium(self, sample_df):
        result = add_domain_features(sample_df)
        expected = sample_df['Number of vessels fluro'] * sample_df['Thallium']
        pd.testing.assert_series_equal(result['Vessel_Thallium'], expected, check_names=False)

    def test_angina_st(self, sample_df):
        result = add_domain_features(sample_df)
        expected = sample_df['Exercise angina'] * sample_df['ST depression']
        pd.testing.assert_series_equal(result['Angina_ST'], expected, check_names=False)

    def test_cardiac_risk_range(self, sample_df):
        result = add_domain_features(sample_df)
        assert result['Cardiac_Risk'].between(0, 5).all()

    def test_binary_flags(self, sample_df):
        result = add_domain_features(sample_df)
        for col in ['Typical_Angina', 'Has_Vessel', 'Thallium_Abnormal']:
            assert set(result[col].unique()).issubset({0, 1}), f"{col} not binary"

    def test_typical_angina_values(self, sample_df):
        result = add_domain_features(sample_df)
        # Rows 0 and 3 have Chest pain type == 4
        assert result.loc[0, 'Typical_Angina'] == 1
        assert result.loc[1, 'Typical_Angina'] == 0
        assert result.loc[3, 'Typical_Angina'] == 1

    def test_thallium_abnormal_values(self, sample_df):
        result = add_domain_features(sample_df)
        # Rows 0,2 have Thallium==3 (normal), rows 1,3 have 7,6 (abnormal)
        assert result.loc[0, 'Thallium_Abnormal'] == 0
        assert result.loc[1, 'Thallium_Abnormal'] == 1
        assert result.loc[2, 'Thallium_Abnormal'] == 0
        assert result.loc[3, 'Thallium_Abnormal'] == 1

    def test_get_domain_features_count(self):
        assert len(get_domain_features()) == 18


if __name__ == "__main__":
    test_original_stats()
