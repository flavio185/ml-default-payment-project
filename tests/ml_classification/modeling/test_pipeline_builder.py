"""Tests for pipeline builder module."""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_classification.modeling.pipeline_builder import (
    create_sklearn_pipeline,
    get_pipeline_info,
)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 60],
            "limit_bal": [10000, 20000, 30000, 40000],
            "sex": ["male", "female", "male", "female"],
            "education": ["university", "university", "graduate", "high_school"],
            "bill_amt1": [1000, 2000, 3000, 4000],
        }
    )


def test_create_sklearn_pipeline_with_logistic_regression(sample_data):
    """Test pipeline creation with LogisticRegression."""
    model = LogisticRegression(random_state=42)
    pipeline = create_sklearn_pipeline(sample_data, model)

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "preprocessor"
    assert pipeline.steps[1][0] == "classifier"
    assert isinstance(pipeline.steps[1][1], LogisticRegression)


def test_create_sklearn_pipeline_with_random_forest(sample_data):
    """Test pipeline creation with RandomForest."""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    pipeline = create_sklearn_pipeline(sample_data, model)

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.steps[1][1], RandomForestClassifier)


def test_get_pipeline_info(sample_data):
    """Test extracting pipeline information."""
    model = LogisticRegression(random_state=42)
    pipeline = create_sklearn_pipeline(sample_data, model)

    # Create mock target
    y = pd.Series([0, 1, 0, 1])

    # Fit pipeline to populate attributes
    pipeline.fit(sample_data, y)

    info = get_pipeline_info(pipeline)

    assert "steps" in info
    assert "classifier" in info
    assert "transformers" in info
    assert info["steps"] == ["preprocessor", "classifier"]
    assert info["classifier"] == "LogisticRegression"
    assert "cat" in info["transformers"]
    assert "num" in info["transformers"]


def test_pipeline_can_be_fitted(sample_data):
    """Test that created pipeline can be fitted."""
    model = LogisticRegression(random_state=42)
    pipeline = create_sklearn_pipeline(sample_data, model)

    y = pd.Series([0, 1, 0, 1])
    pipeline.fit(sample_data, y)

    # Should be able to predict after fitting
    predictions = pipeline.predict(sample_data)
    assert len(predictions) == len(sample_data)
    assert set(predictions).issubset({0, 1})
