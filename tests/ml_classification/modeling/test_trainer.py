"""Tests for trainer module."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_classification.modeling.pipeline_builder import create_sklearn_pipeline
from ml_classification.modeling.trainer import train_and_evaluate, train_model


@pytest.fixture
def sample_train_data():
    """Create sample training data."""
    X_train = pd.DataFrame(
        {
            "age": [25, 35, 45, 60, 28, 33, 48, 55],
            "limit_bal": [10000, 20000, 30000, 40000, 15000, 25000, 35000, 45000],
            "sex": ["male", "female", "male", "female", "male", "female", "male", "female"],
            "bill_amt1": [1000, 2000, 3000, 4000, 1500, 2500, 3500, 4500],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    return X_train, y_train


@pytest.fixture
def sample_test_data():
    """Create sample test data."""
    X_test = pd.DataFrame(
        {
            "age": [30, 40],
            "limit_bal": [18000, 28000],
            "sex": ["male", "female"],
            "bill_amt1": [1800, 2800],
        }
    )
    y_test = pd.Series([0, 1])
    return X_test, y_test


def test_train_model(sample_train_data):
    """Test model training."""
    X_train, y_train = sample_train_data
    model = LogisticRegression(random_state=42, max_iter=1000)
    pipeline = create_sklearn_pipeline(X_train, model)

    trained_pipeline = train_model(pipeline, X_train, y_train)

    assert isinstance(trained_pipeline, Pipeline)
    # Check that pipeline is fitted
    assert hasattr(trained_pipeline, "classes_")


def test_train_and_evaluate(sample_train_data, sample_test_data):
    """Test training and evaluation."""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data

    model = LogisticRegression(random_state=42, max_iter=1000)
    pipeline = create_sklearn_pipeline(X_train, model)

    trained_pipeline, metrics, cm, y_proba = train_and_evaluate(
        pipeline, X_train, y_train, X_test, y_test
    )

    # Check pipeline is trained
    assert isinstance(trained_pipeline, Pipeline)
    assert hasattr(trained_pipeline, "classes_")

    # Check metrics
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "avg_precision" in metrics

    # Check values are valid
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1

    # Check confusion matrix exists
    assert cm is not None

    # Check probabilities
    assert len(y_proba) == len(X_test)
    assert all(0 <= p <= 1 for p in y_proba)


def test_trained_model_can_predict(sample_train_data, sample_test_data):
    """Test that trained model can make predictions."""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data

    model = LogisticRegression(random_state=42, max_iter=1000)
    pipeline = create_sklearn_pipeline(X_train, model)

    trained_pipeline, _, _, _ = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)

    # Make predictions
    predictions = trained_pipeline.predict(X_test)
    probabilities = trained_pipeline.predict_proba(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})
    assert probabilities.shape == (len(X_test), 2)
