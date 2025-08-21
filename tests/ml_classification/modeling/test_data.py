import pandas as pd
import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ml_classification.modeling.data import build_preprocessor
from unittest.mock import patch, MagicMock
from ml_classification.modeling.data import load_data

def test_build_preprocessor_categorizes_columns_correctly():
    df = pd.DataFrame({
        "age": [25, 35, 45],
        "income": [50000.0, 60000.0, 70000.0],
        "gender": ["M", "F", "M"],
        "education": pd.Series(["Bachelors", "Masters", "PhD"], dtype="category")
    })
    preprocessor = build_preprocessor(df)
    # Check that the transformer is a ColumnTransformer
    assert isinstance(preprocessor, ColumnTransformer)
    # Extract transformer names and columns
    transformers = dict((name, cols) for name, _, cols in preprocessor.transformers)
    # Categorical columns
    assert set(transformers["cat"]) == {"gender", "education"}
    # Numerical columns
    assert set(transformers["num"]) == {"age", "income"}

def test_build_preprocessor_transform_output_shape():
    df = pd.DataFrame({
        "age": [25, 35, 45],
        "income": [50000.0, 60000.0, 70000.0],
        "gender": ["M", "F", "M"],
    })
    preprocessor = build_preprocessor(df)
    transformed = preprocessor.fit_transform(df)
    # gender has 2 unique values -> 2 one-hot columns, age and income are numerical
    assert transformed.shape[1] == 2 + 2  # 2 one-hot + 2 numerical

def test_build_preprocessor_handles_no_categorical_columns():
    df = pd.DataFrame({
        "age": [25, 35, 45],
        "income": [50000.0, 60000.0, 70000.0],
    })
    preprocessor = build_preprocessor(df)
    # Should only have the numerical transformer
    transformers = dict((name, cols) for name, _, cols in preprocessor.transformers)
    assert transformers["cat"] == []
    assert set(transformers["num"]) == {"age", "income"}

def test_build_preprocessor_handles_no_numerical_columns():
    df = pd.DataFrame({
        "gender": ["M", "F", "M"],
        "education": pd.Series(["Bachelors", "Masters", "PhD"], dtype="category")
    })
    preprocessor = build_preprocessor(df)
    transformers = dict((name, cols) for name, _, cols in preprocessor.transformers)
    assert set(transformers["cat"]) == {"gender", "education"}
    assert transformers["num"] == []

def test_build_preprocessor_handles_empty_dataframe():
    df = pd.DataFrame()
    preprocessor = build_preprocessor(df)
    transformers = dict((name, cols) for name, _, cols in preprocessor.transformers)
    assert transformers["cat"] == []
    assert transformers["num"] == []
@pytest.fixture
def sample_parquet_df():
    data = {
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "default_payment_next_month": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)

@patch("ml_classification.modeling.data.pd.read_parquet")
@patch("ml_classification.modeling.data.train_test_split")
def test_load_data_reads_parquet_and_splits(mock_split, mock_read_parquet, sample_parquet_df):
    mock_read_parquet.return_value = sample_parquet_df
    # Simulate train_test_split output
    X = sample_parquet_df.drop(columns=["default_payment_next_month"])
    y = sample_parquet_df["default_payment_next_month"]
    split_result = ("X_train", "X_test", "y_train", "y_test")
    mock_split.return_value = split_result

    result = load_data("dummy_path.parquet")

    mock_read_parquet.assert_called_once_with("dummy_path.parquet")
    mock_split.assert_called_once()
    # Check that X and y passed to train_test_split are correct
    args, kwargs = mock_split.call_args
    pd.testing.assert_frame_equal(args[0], X)
    pd.testing.assert_series_equal(args[1], y)
    assert kwargs["test_size"] == 0.2
    assert kwargs["random_state"] == 42
    assert kwargs["stratify"].equals(y)
    assert result == split_result

@patch("ml_classification.modeling.data.pd.read_parquet")
def test_load_data_raises_if_target_missing(mock_read_parquet, sample_parquet_df):
    df_no_target = sample_parquet_df.drop(columns=["default_payment_next_month"])
    mock_read_parquet.return_value = df_no_target
    with pytest.raises(KeyError):
        load_data("dummy_path.parquet")
