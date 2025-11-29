"""Preprocessing and transformation utilities.

This module contains preprocessing logic including scaling, encoding,
and preprocessing configuration management.
"""

from datetime import datetime
import json
from pathlib import Path

import boto3
from loguru import logger
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for preprocessing data.

    Args:
        X: DataFrame with features to preprocess

    Returns:
        Fitted ColumnTransformer with categorical and numerical transformers
    """
    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype == "category"]
    numerical_cols = [c for c in X.columns if X[c].dtype in ["int64", "float64"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )
    return preprocessor


def get_preprocessing_config(X: pd.DataFrame) -> dict:
    """Generate preprocessing configuration from DataFrame.

    Args:
        X: DataFrame with features

    Returns:
        Dictionary with preprocessing configuration
    """
    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype == "category"]
    numerical_cols = [c for c in X.columns if X[c].dtype in ["int64", "float64"]]

    config = {
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
        "total_features": len(X.columns),
        "created_at": datetime.utcnow().isoformat(),
    }
    return config


def save_preprocessing_config(config: dict, output_path: str | Path) -> None:
    """Save preprocessing configuration to JSON file.

    Args:
        config: Preprocessing configuration dictionary
        output_path: Path to save configuration (local or S3)
    """
    output_path_str = str(output_path)

    if output_path_str.startswith("s3://"):
        # Save to S3
        bucket = output_path_str.split("/")[2]
        key = "/".join(output_path_str.split("/")[3:])
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(config, indent=2),
            ContentType="application/json",
        )
        logger.info(f"Preprocessing config saved to S3: {output_path_str}")
    else:
        # Save locally
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Preprocessing config saved locally: {output_path}")


def load_preprocessing_config(config_path: str | Path) -> dict:
    """Load preprocessing configuration from JSON file.

    Args:
        config_path: Path to configuration file (local or S3)

    Returns:
        Preprocessing configuration dictionary
    """
    config_path_str = str(config_path)

    if config_path_str.startswith("s3://"):
        # Load from S3
        bucket = config_path_str.split("/")[2]
        key = "/".join(config_path_str.split("/")[3:])
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket, Key=key)
        config = json.loads(response["Body"].read())
        logger.info(f"Preprocessing config loaded from S3: {config_path_str}")
    else:
        # Load locally
        with open(config_path) as f:
            config = json.load(f)
        logger.info(f"Preprocessing config loaded locally: {config_path}")

    return config


def build_preprocessor_from_config(config: dict) -> ColumnTransformer:
    """Build preprocessor from saved configuration.

    Args:
        config: Preprocessing configuration dictionary

    Returns:
        ColumnTransformer configured according to the config
    """
    categorical_cols = config["categorical_columns"]
    numerical_cols = config["numerical_columns"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )
    return preprocessor
