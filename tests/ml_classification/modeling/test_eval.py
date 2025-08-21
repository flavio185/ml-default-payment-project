import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from ml_classification.modeling.eval import evaluate_model

@pytest.fixture
def dummy_data():
    # Simple binary classification data
    X_test = np.array([[0], [1], [2], [3], [4]])
    y_test = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.6, 0.4], [0.3, 0.7]])
    return X_test, y_test, y_pred, y_proba

@pytest.fixture
def dummy_model(dummy_data):
    X_test, y_test, y_pred, y_proba = dummy_data
    model = MagicMock()
    model.predict.return_value = y_pred
    model.predict_proba.return_value = y_proba
    return model

@patch("ml_classification.modeling.eval.logger")
@patch("ml_classification.modeling.eval.plot_confusion_matrix")
def test_evaluate_model_basic(mock_plot_cm, mock_logger, dummy_model, dummy_data):
    X_test, y_test, y_pred, y_proba = dummy_data
    mock_plot_cm.return_value = np.array([[2, 0], [1, 2]])

    metrics, cm, y_proba_out = evaluate_model(dummy_model, X_test, y_test)

    # Check returned metrics keys
    expected_keys = {
        "accuracy", "roc_auc", "avg_precision", "f1", "log_loss",
        "precision_at_1", "recall_at_1",
        "precision_at_5", "recall_at_5",
        "precision_at_10", "recall_at_10",
        "precision_at_20", "recall_at_20"
    }
    assert expected_keys.issubset(metrics.keys())
    # Check confusion matrix is returned as expected
    assert (cm == np.array([[2, 0], [1, 2]])).all()
    # Check y_proba_out shape
    assert np.allclose(y_proba_out, y_proba[:, 1])
    # Check logger was called (at least for metrics and report)
    assert mock_logger.info.call_count > 2

@patch("ml_classification.modeling.eval.logger")
@patch("ml_classification.modeling.eval.plot_confusion_matrix")
def test_evaluate_model_handles_perfect_model(mock_plot_cm, mock_logger):
    # Perfect predictions
    X_test = np.array([[0], [1], [2]])
    y_test = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 1])
    y_proba = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    model = MagicMock()
    model.predict.return_value = y_pred
    model.predict_proba.return_value = y_proba
    mock_plot_cm.return_value = np.array([[1, 0], [0, 2]])

    metrics, cm, y_proba_out = evaluate_model(model, X_test, y_test)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert (cm == np.array([[1, 0], [0, 2]])).all()
    assert np.allclose(y_proba_out, y_proba[:, 1])

@patch("ml_classification.modeling.eval.logger")
@patch("ml_classification.modeling.eval.plot_confusion_matrix")
def test_evaluate_model_handles_all_zero_predictions(mock_plot_cm, mock_logger):
    # All predictions are zero
    X_test = np.array([[0], [1], [2]])
    y_test = np.array([0, 1, 1])
    y_pred = np.array([0, 0, 0])
    y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    model = MagicMock()
    model.predict.return_value = y_pred
    model.predict_proba.return_value = y_proba
    mock_plot_cm.return_value = np.array([[1, 0], [2, 0]])

    metrics, cm, y_proba_out = evaluate_model(model, X_test, y_test)
    assert metrics["accuracy"] == pytest.approx(1/3)
    assert (cm == np.array([[1, 0], [2, 0]])).all()
    assert np.allclose(y_proba_out, y_proba[:, 1])