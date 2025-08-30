import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.metrics import confusion_matrix
from ml_classification.modeling.eval import precision_recall_at_k, plot_confusion_matrix, evaluate_model

@pytest.fixture
def y_true():
    return np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1])

@pytest.fixture
def y_scores():
    return np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.4, 0.95, 0.05, 0.85])

@pytest.fixture
def y_pred():
    return (np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.4, 0.95, 0.05, 0.85]) > 0.5).astype(int)

def test_precision_recall_at_k(y_true, y_scores):
    precision, recall = precision_recall_at_k(y_true, y_scores, k=0.2)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1

@patch("ml_classification.modeling.eval.plt.figure")
@patch("ml_classification.modeling.eval.sns.heatmap")
def test_plot_confusion_matrix(mock_heatmap, mock_figure, y_true, y_pred):
    cm = plot_confusion_matrix(y_true, y_pred)
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (2, 2)
    mock_heatmap.assert_called_once()

def test_evaluate_model(y_true, y_pred):
    # Create mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = y_pred
    mock_model.predict_proba.return_value = np.vstack([1 - y_pred, y_pred]).T

    # Create dummy X_test
    X_test = pd.DataFrame(np.random.rand(len(y_true), 5))

    metrics, cm, y_proba = evaluate_model(mock_model, X_test, y_true)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert isinstance(cm, np.ndarray)
    assert isinstance(y_proba, np.ndarray)
    assert len(y_proba) == len(y_true)
