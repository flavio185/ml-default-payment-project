"""Tests for feature engineering functions."""

import pandas as pd
import pytest

from ml_classification.features.engineering import (
    create_age_bins,
    create_bill_trend,
    create_pay_ratio,
    create_utilization,
    engineer_features,
    get_feature_names,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 60],
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
            "limit_bal": [10000, 20000, 30000, 40000],
        }
    )


def test_create_age_bins(sample_data):
    """Test age binning feature creation."""
    result = create_age_bins(sample_data.copy())
    assert "age_bin" in result.columns
    assert result["age_bin"].iloc[0] == "18_25"
    assert result["age_bin"].iloc[1] == "26_35"
    assert result["age_bin"].iloc[2] == "36_50"
    assert result["age_bin"].iloc[3] == "50_plus"


def test_create_bill_trend(sample_data):
    """Test bill trend feature creation."""
    result = create_bill_trend(sample_data.copy())
    assert "bill_trend" in result.columns
    assert result["bill_trend"].iloc[0] == 500  # 1500 - 1000


def test_create_pay_ratio(sample_data):
    """Test pay ratio feature creation."""
    result = create_pay_ratio(sample_data.copy())
    assert "pay_ratio" in result.columns
    assert result["pay_ratio"].iloc[0] > 0
    assert result["pay_ratio"].iloc[0] < 1  # pays less than total bill


def test_create_utilization(sample_data):
    """Test utilization feature creation."""
    result = create_utilization(sample_data.copy())
    assert "utilization" in result.columns
    assert result["utilization"].iloc[0] == 0.15  # 1500 / 10000


def test_engineer_features(sample_data):
    """Test full feature engineering pipeline."""
    result = engineer_features(sample_data.copy())
    feature_names = get_feature_names()

    # Check all engineered features are present
    for feature in feature_names:
        assert feature in result.columns

    # Check data types
    assert result["age_bin"].dtype == "category"
    assert result["bill_trend"].dtype in ["int64", "float64"]
    assert result["pay_ratio"].dtype == "float64"
    assert result["utilization"].dtype == "float64"


def test_get_feature_names():
    """Test getting feature names."""
    feature_names = get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) == 4
    assert "age_bin" in feature_names
    assert "bill_trend" in feature_names
    assert "pay_ratio" in feature_names
    assert "utilization" in feature_names
