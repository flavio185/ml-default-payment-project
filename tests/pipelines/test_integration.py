"""Integration tests for pipelines.

These tests verify that the pipelines work together correctly.
"""

import pandas as pd
import pytest

from ml_classification.features.engineering import engineer_features
from ml_classification.features.preprocessing import build_preprocessor


@pytest.fixture
def sample_raw_data():
    """Create sample raw data similar to Silver layer."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 60],
            "limit_bal": [10000, 20000, 30000, 40000],
            "sex": [1, 2, 1, 2],
            "education": [1, 2, 3, 4],
            "marriage": [1, 2, 1, 2],
            "bill_amt1": [1000, 2000, 3000, 4000],
            "bill_amt2": [1100, 2100, 3100, 4100],
            "bill_amt3": [1200, 2200, 3200, 4200],
            "bill_amt4": [1300, 2300, 3300, 4300],
            "bill_amt5": [1400, 2400, 3400, 4400],
            "bill_amt6": [1500, 2500, 3500, 4500],
            "pay_amt1": [500, 1000, 1500, 2000],
            "pay_amt2": [500, 1000, 1500, 2000],
            "pay_amt3": [500, 1000, 1500, 2000],
            "pay_amt4": [500, 1000, 1500, 2000],
            "pay_amt5": [500, 1000, 1500, 2000],
            "pay_amt6": [500, 1000, 1500, 2000],
            "default_payment_next_month": [0, 1, 0, 1],
        }
    )


def test_feature_to_training_pipeline_integration(sample_raw_data):
    """Test that features created can be used for training preprocessing."""
    # Simulate feature pipeline
    df_features = engineer_features(sample_raw_data.copy())

    # Simulate training pipeline - drop target
    X = df_features.drop(columns=["default_payment_next_month"], errors="ignore")

    # Build preprocessor (as done in training)
    preprocessor = build_preprocessor(X)

    # Should be able to fit and transform
    X_transformed = preprocessor.fit_transform(X)

    # Check output shape
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] > 0  # Should have some features after transformation


def test_feature_consistency_train_inference(sample_raw_data):
    """Test that feature engineering produces consistent results."""
    # First run (simulate training)
    df_train = engineer_features(sample_raw_data.copy())

    # Second run (simulate inference)
    df_inference = engineer_features(sample_raw_data.copy())

    # Should produce identical results
    pd.testing.assert_frame_equal(df_train, df_inference)


def test_engineered_features_present(sample_raw_data):
    """Test that all engineered features are created."""
    df_features = engineer_features(sample_raw_data.copy())

    # Check engineered features exist
    assert "age_bin" in df_features.columns
    assert "bill_trend" in df_features.columns
    assert "pay_ratio" in df_features.columns
    assert "utilization" in df_features.columns

    # Check no null values introduced
    assert df_features["age_bin"].notna().all()
    assert df_features["bill_trend"].notna().all()
    assert df_features["pay_ratio"].notna().all()
    assert df_features["utilization"].notna().all()
