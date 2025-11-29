"""Feature Pipeline - Orchestrates feature engineering from Silver to Gold layer.

This pipeline can run independently and on its own schedule (e.g., daily).
It creates versioned features that are consumed by the training pipeline.
"""

from datetime import datetime, timezone
import json

import boto3
from data_processing.check_s3 import wait_for_s3_object
from loguru import logger
import pandas as pd
import typer

from ml_classification.config import S3_BUCKET
from ml_classification.features.engineering import engineer_features, get_feature_names
from ml_classification.features.preprocessing import (
    get_preprocessing_config,
    save_preprocessing_config,
)

app = typer.Typer()


def get_dataset_metadata(data_path: str) -> dict:
    """Get metadata for the source dataset from S3.

    Args:
        data_path: S3 path to the dataset

    Returns:
        Dictionary with dataset metadata
    """
    bucket = data_path.split("/")[2]
    key = "/".join(data_path.split("/")[3:])
    s3 = boto3.client("s3")

    # Get object versions
    versions = s3.list_object_versions(Bucket=bucket, Prefix=key)
    latest_version = versions["Versions"][0]  # Assumes latest is first

    metadata = {
        "source_uri": data_path,
        "version_id": latest_version["VersionId"],
        "last_modified": latest_version["LastModified"].isoformat(),
        "size_bytes": latest_version["Size"],
    }
    return metadata


def save_feature_metadata(
    feature_metadata: dict, output_path: str, metadata_suffix: str = "_metadata.json"
) -> None:
    """Save feature metadata to S3 alongside the feature data.

    Args:
        feature_metadata: Dictionary with feature metadata
        output_path: S3 path where features are saved
        metadata_suffix: Suffix for metadata file
    """
    # Create metadata path by replacing .parquet with _metadata.json
    metadata_path = output_path.replace(".parquet", metadata_suffix)

    bucket = metadata_path.split("/")[2]
    key = "/".join(metadata_path.split("/")[3:])

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(feature_metadata, indent=2),
        ContentType="application/json",
    )
    logger.info(f"Feature metadata saved to: {metadata_path}")


@app.command()
def run_feature_pipeline(
    input_path: str = "s3://" + S3_BUCKET + "/silver/credit_card_default.parquet",
    output_path: str = "s3://" + S3_BUCKET + "/gold/credit_card_default_features.parquet",
    feature_version: str = None,
):
    """Run the feature engineering pipeline.

    Args:
        input_path: Path to input data (Silver layer)
        output_path: Path to save features (Gold layer)
        feature_version: Optional feature version string (defaults to timestamp)
    """
    logger.info("=" * 60)
    logger.info("FEATURE PIPELINE STARTED")
    logger.info("=" * 60)

    # Generate feature version if not provided
    if feature_version is None:
        feature_version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info(f"Feature version: {feature_version}")

    # Wait for input data and load
    logger.info(f"Loading Silver dataset from: {input_path}")
    wait_for_s3_object(S3_BUCKET, "silver/credit_card_default.parquet", timeout=60)
    df = pd.read_parquet(input_path, storage_options={"anon": False})
    logger.info(f"Loaded {len(df)} rows from Silver layer")

    # Get source dataset metadata
    source_metadata = get_dataset_metadata(input_path)

    # Apply feature engineering
    df_features = engineer_features(df)
    logger.info(f"Created {len(get_feature_names())} engineered features")

    # Generate preprocessing configuration
    # Drop target if present for preprocessing config
    X = df_features.drop(columns=["default_payment_next_month"], errors="ignore")
    preprocessing_config = get_preprocessing_config(X)

    # Create comprehensive feature metadata
    feature_metadata = {
        "feature_version": feature_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": source_metadata,
        "total_rows": len(df_features),
        "total_columns": len(df_features.columns),
        "feature_columns": list(df_features.columns),
        "engineered_features": get_feature_names(),
        "preprocessing_config": preprocessing_config,
    }

    # Save features to Gold layer
    logger.info(f"Saving Gold dataset to {output_path}...")
    df_features.to_parquet(output_path, index=False, storage_options={"anon": False})
    logger.success(f"Features saved successfully: {len(df_features)} rows")

    # Save feature metadata
    save_feature_metadata(feature_metadata, output_path)

    # Also save preprocessing config separately for easy access
    preprocessing_config_path = output_path.replace(".parquet", "_preprocessing_config.json")
    save_preprocessing_config(preprocessing_config, preprocessing_config_path)

    logger.info("=" * 60)
    logger.success("FEATURE PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Feature version: {feature_version}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Total features: {len(df_features.columns)}")
    logger.info(f"Engineered features: {', '.join(get_feature_names())}")


if __name__ == "__main__":
    app()
