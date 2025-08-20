import pytest
from ml_classification.modeling.train import main as train
import pandas as pd

def test_main_runs(tmp_path):
    # Create a dummy dataset
    df = pd.DataFrame({
        'default_payment_next_month': [0, 1, 0, 1],
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8]
    })
    features_path = tmp_path/"dataset.csv"
    df.to_parquet(features_path, index=False)
    train(input_path=features_path)
    # No assertion, just check code runs
