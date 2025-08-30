from pathlib import Path
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from ml_classification.modeling.data import build_preprocessor, get_table_metadata, load_data

# Sample DataFrame for testing
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 45],
        "gender": ["male", "female", "female"],
        "income": [50000, 60000, 80000],
        "default_payment_next_month": [0, 1, 0],
        "ingestion_time": pd.Timestamp.now()
    })

def test_build_preprocessor(sample_df):
    X = sample_df.drop(columns=["default_payment_next_month", "ingestion_time"])
    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == X.shape[0]

@patch("ml_classification.modeling.data.boto3.client")
@patch("ml_classification.modeling.data.open", new_callable=mock_open)
@patch("ml_classification.modeling.data.REPORTS_DIR", new_callable=lambda: Path("/tmp"))
def test_get_table_metadata(mock_reports_dir, mock_open_file, mock_boto_client):
    mock_s3 = MagicMock()
    mock_s3.list_object_versions.return_value = {
        "Versions": [{
            "VersionId": "abc123",
            "LastModified": pd.Timestamp("2025-08-30"),
            "Size": 123456,
            "StorageClass": "STANDARD"
        }]
    }
    mock_boto_client.return_value = mock_s3

    get_table_metadata(test_size=0.2, random_state=42)
    mock_open_file.assert_called_once()

@patch("ml_classification.modeling.data.pd.read_parquet")
@patch("ml_classification.modeling.data.get_table_metadata")
def test_load_data(mock_metadata, mock_read_parquet, sample_df):
    mock_read_parquet.return_value = sample_df
    X_train, X_test, y_train, y_test = load_data(test_size=0.5, random_state=1)

    assert len(X_train) + len(X_test) == len(sample_df)
    assert set(y_train).union(set(y_test)) == set(sample_df["default_payment_next_month"])
