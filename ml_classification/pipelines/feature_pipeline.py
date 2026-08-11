"""Feature Pipeline - Orchestrates feature engineering from Silver to Gold layer.

This pipeline can run independently and on its own schedule (e.g., daily).
It creates versioned features that are consumed by the training pipeline.
"""

from datetime import datetime, timezone

from loguru import logger
from mlops_toolkit.io import get_dataset_metadata, save_feature_metadata, wait_for_s3_object
import pandas as pd
import typer

from ml_classification.config import FEAST_REPO_PATH, S3_BUCKET
from ml_classification.features.engineering import engineer_features, get_feature_names
from ml_classification.features.preprocessing import (
    get_preprocessing_config,
    save_preprocessing_config,
)

app = typer.Typer()


def _feast_materialize():
    """Registra feature definitions e materializa features para o online store."""
    try:
        from feast import FeatureStore
        from feast.features.credit_card_features import (
            credit_card_features,
            customer,
            gold_source,
        )

        store = FeatureStore(repo_path=str(FEAST_REPO_PATH))
        store.apply([customer, credit_card_features, gold_source])
        store.materialize_incremental(end_date=datetime.now(timezone.utc))
        logger.success("Feast: features registradas e materializadas com sucesso")
    except Exception as e:
        logger.warning(f"Feast materialization skipped: {e}")


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

    # Capture this write's own S3 version so a later drift check (or anything else)
    # can fetch this *exact* snapshot training will read even after output_path has
    # been overwritten by subsequent runs -- source_dataset above only pins the
    # Silver input, not this.
    feature_metadata["training_dataset"] = get_dataset_metadata(output_path)

    # Save feature metadata
    save_feature_metadata(feature_metadata, output_path)

    # Also save preprocessing config separately for easy access
    preprocessing_config_path = output_path.replace(".parquet", "_preprocessing_config.json")
    save_preprocessing_config(preprocessing_config, preprocessing_config_path)

    # Register and materialize features to Feast online store
    _feast_materialize()

    logger.info("=" * 60)
    logger.success("FEATURE PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Feature version: {feature_version}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Total features: {len(df_features.columns)}")
    logger.info(f"Engineered features: {', '.join(get_feature_names())}")


if __name__ == "__main__":
    app()
