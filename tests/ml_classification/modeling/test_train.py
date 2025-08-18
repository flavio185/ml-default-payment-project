import pytest
from ml_classification.modeling import train
import pandas as pd

def test_main_runs(tmp_path):
    # Create a dummy dataset
    df = pd.DataFrame({
        'default': [0, 1, 0, 1],
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8]
    })
    features_path = tmp_path/"dataset.csv"
    df.to_csv(features_path, index=False)
    train.main(features_path=features_path)
    # No assertion, just check code runs
