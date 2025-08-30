from datetime import datetime

import boto3
from loguru import logger
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import typer

from ml_classification.config import REPORTS_DIR, S3_BUCKET

app = typer.Typer()


def build_preprocessor(X):
    """Builds a ColumnTransformer for preprocessing data."""
    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype == "category"]
    numerical_cols = [c for c in X.columns if X[c].dtype in ["int64", "float64"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )
    return preprocessor


def get_table_metadata(test_size, random_state):
    """Gets metadata for the S3 object."""
    # S3 object details
    bucket = S3_BUCKET
    key = "gold/credit_card_default_features.parquet"
    # Initialize S3 client
    s3 = boto3.client("s3")

    # Get object versions
    versions = s3.list_object_versions(Bucket=bucket, Prefix=key)
    latest_version = versions["Versions"][0]  # Assumes latest is first

    # Build metadata dictionary
    metadata = {
        "s3_uri": f"s3://{S3_BUCKET}/{key}",
        "version_id": latest_version["VersionId"],
        "last_modified": latest_version["LastModified"].isoformat(),
        "size_bytes": latest_version["Size"],
        "storage_class": latest_version.get("StorageClass", "STANDARD"),
        "logged_at": datetime.utcnow().isoformat(),
        "split_strategy": {
            "method": "train_test_split",
            "test_size": test_size,
            "random_state": random_state,
        },
    }
    # save metadata to a json file
    with open(REPORTS_DIR / "s3_metadata.json", "w") as f:
        import json

        json.dump(metadata, f, indent=4)


@app.command()
def load_data(test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """Loads data from S3 and splits it into training and testing sets."""
    data_path = "s3://" + S3_BUCKET + "/gold/credit_card_default_features.parquet"
    get_table_metadata(test_size, random_state)
    target_col = "default_payment_next_month"
    logger.info(f"Loading dataset from: {data_path}")

    df = pd.read_parquet(data_path, storage_options={"anon": False})
    X = df.drop(columns=[target_col, "ingestion_time"], errors="ignore")
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info("Data loading and splitting completed.")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    app()
