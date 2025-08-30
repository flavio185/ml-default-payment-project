import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from ml_classification.modeling.models import (
    logistic_regression_model,
    svm_model,
    random_forest_model,
)

def test_logistic_regression_model():
    model = logistic_regression_model()
    assert isinstance(model, LogisticRegression)
    assert model.C == 1.0
    assert model.max_iter == 100
    assert model.solver == "liblinear"

def test_svm_model():
    model = svm_model()
    assert isinstance(model, SVC)
    assert model.C == 0.1
    assert model.kernel == "linear"
    assert model.probability is True

def test_random_forest_model():
    model = random_forest_model()
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert model.max_depth == 5
    assert model.random_state == 42
