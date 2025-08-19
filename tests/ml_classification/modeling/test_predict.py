import pytest
from ml_classification.modeling import predict
from pathlib import Path

def test_main_runs(tmp_path):
    features_path = tmp_path/"test_features.csv"
    model_path = tmp_path/"model.pkl"
    predictions_path = tmp_path/"test_predictions.csv"
    features_path.write_text("col1,col2\n1,2\n3,4\n")
    model_path.write_bytes(b"dummy")
    predict.main(features_path=features_path, model_path=model_path, predictions_path=predictions_path)
    # No assertion, just check code runs
