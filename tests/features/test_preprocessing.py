"""Tests for preprocessing functions."""

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from ml_classification.features.preprocessing import (
    build_preprocessor,
    build_preprocessor_from_config,
    get_preprocessing_config,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 60],
            "limit_bal": [10000, 20000, 30000, 40000],
            "sex": ["male", "female", "male", "female"],
            "education": ["university", "university", "graduate", "high_school"],
        }
    )


def test_build_preprocessor(sample_data):
    """Test building preprocessor from data."""
    preprocessor = build_preprocessor(sample_data)

    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) == 2

    # Check transformer names
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "cat" in transformer_names
    assert "num" in transformer_names


def test_get_preprocessing_config(sample_data):
    """Test generating preprocessing configuration."""
    config = get_preprocessing_config(sample_data)

    assert "categorical_columns" in config
    assert "numerical_columns" in config
    assert "total_features" in config
    assert "created_at" in config

    # Check columns are correctly categorized
    assert "sex" in config["categorical_columns"]
    assert "education" in config["categorical_columns"]
    assert "age" in config["numerical_columns"]
    assert "limit_bal" in config["numerical_columns"]

    assert config["total_features"] == 4


def test_build_preprocessor_from_config(sample_data):
    """Test building preprocessor from saved configuration."""
    config = get_preprocessing_config(sample_data)
    preprocessor = build_preprocessor_from_config(config)

    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) == 2

    # Check that the same columns are used
    cat_transformer = preprocessor.transformers[0]
    num_transformer = preprocessor.transformers[1]

    assert cat_transformer[2] == config["categorical_columns"]
    assert num_transformer[2] == config["numerical_columns"]
