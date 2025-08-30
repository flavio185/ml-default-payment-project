import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_classification.modeling.train import create_pipeline, log_model_run

@pytest.fixture
def X_train():
    return pd.DataFrame(np.random.rand(10, 4), columns=["A", "B", "C", "D"])

@pytest.fixture
def X_test():
    return pd.DataFrame(np.random.rand(5, 4), columns=["A", "B", "C", "D"])

@pytest.fixture
def y_train():
    return np.random.randint(0, 2, size=10)

@pytest.fixture
def y_test():
    return np.random.randint(0, 2, size=5)

def test_create_pipeline(X_train):
    model = LogisticRegression()
    pipeline = create_pipeline(X_train, model)
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in dict(pipeline.named_steps)
    assert "classifier" in dict(pipeline.named_steps)

@patch("ml_classification.modeling.train.mlflow")
@patch("ml_classification.modeling.train.plt")
@patch("ml_classification.modeling.train.infer_signature")
@patch("ml_classification.modeling.train.REPORTS_DIR", autospec=True)
@patch("ml_classification.modeling.train.open", create=True)
def test_log_model_run(
    mock_open,
    mock_reports_dir,
    mock_signature,
    mock_plt,
    mock_mlflow,
    X_train,
    X_test,
):
    # Setup dummy pipeline and metrics
    dummy_model = LogisticRegression()
    pipeline = create_pipeline(X_train, dummy_model)
    pipeline.fit(X_train, np.random.randint(0, 2, size=10))

    metrics = {"accuracy": 0.9, "f1": 0.8}
    cm = np.array([[3, 1], [0, 1]])
    algorithm = "LogisticRegression"

    # Mock metadata file
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = '{"s3_uri": "s3://bucket", "version_id": "v1", "split_strategy": "random"}'
    mock_open.return_value = mock_file

    # Patch REPORTS_DIR path
    mock_reports_dir.__truediv__.return_value = "dummy_path"

    log_model_run(pipeline, X_train, X_test, metrics, cm, algorithm)

    mock_mlflow.log_param.assert_any_call("train_size", X_train.shape[0])
    mock_mlflow.log_metrics.assert_called_once_with(metrics)
    mock_mlflow.sklearn.log_model.assert_called_once()
