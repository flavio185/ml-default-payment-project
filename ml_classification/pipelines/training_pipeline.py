"""Training Pipeline - Orchestrates model training using pre-computed features.

This pipeline loads features from the Gold layer and trains models,
ensuring proper tracking of feature versions and model lineage.
"""

import json

import boto3
from loguru import logger
from matplotlib import pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import typer

from ml_classification.config import S3_BUCKET
from ml_classification.features.preprocessing import build_preprocessor
from ml_classification.modeling.eval import evaluate_model
from ml_classification.modeling.models import logistic_regression_model, random_forest_model

app = typer.Typer()


def load_feature_metadata(features_path: str) -> dict:
    """Load feature metadata from S3.

    Args:
        features_path: S3 path to features parquet file

    Returns:
        Dictionary with feature metadata
    """
    metadata_path = features_path.replace(".parquet", "_metadata.json")
    bucket = metadata_path.split("/")[2]
    key = "/".join(metadata_path.split("/")[3:])

    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    metadata = json.loads(response["Body"].read())

    logger.info(f"Loaded feature metadata: version {metadata.get('feature_version')}")
    return metadata


def load_features_with_metadata(
    features_path: str, target_col: str, test_size: float = 0.2, random_state: int = 42
):
    """Load features from Gold layer with metadata.

    Args:
        features_path: S3 path to features
        target_col: Name of target column
        test_size: Proportion of test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)
    """
    logger.info(f"Loading features from: {features_path}")

    # Load features
    df = pd.read_parquet(features_path, storage_options={"anon": False})
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Load metadata
    feature_metadata = load_feature_metadata(features_path)

    # Split features and target
    X = df.drop(columns=[target_col, "ingestion_time"], errors="ignore")
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Add split information to metadata
    feature_metadata["split_strategy"] = {
        "method": "train_test_split",
        "random_state": random_state,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    logger.info(f"Train set: {len(X_train)} rows, Test set: {len(X_test)} rows")

    return X_train, X_test, y_train, y_test, feature_metadata


def create_pipeline(X_train: pd.DataFrame, model) -> Pipeline:
    """Create sklearn pipeline with preprocessing and model.

    Args:
        X_train: Training features for building preprocessor
        model: Sklearn model instance

    Returns:
        Pipeline with preprocessor and model
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", model),
        ]
    )
    return pipeline


def log_model_run(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    metrics: dict,
    cm,
    algorithm: str,
    feature_metadata: dict,
):
    """Log model run to MLflow with all artifacts and metadata.

    Args:
        pipeline: Trained sklearn pipeline
        X_train: Training features
        X_test: Test features
        metrics: Evaluation metrics dictionary
        cm: Confusion matrix plot
        algorithm: Model algorithm name
        feature_metadata: Feature metadata dictionary
    """
    # Log parameters
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    mlflow.log_param("algorithm", algorithm)

    # Log feature metadata
    mlflow.log_param("feature_version", feature_metadata.get("feature_version", "unknown"))
    mlflow.log_param("total_features", feature_metadata.get("total_columns"))
    mlflow.log_param(
        "engineered_features_count", len(feature_metadata.get("engineered_features", []))
    )

    # Log all metrics
    mlflow.log_metrics(metrics)

    # Log confusion matrix plot
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Log feature metadata as artifact
    mlflow.log_dict(feature_metadata, "feature_metadata.json")

    # Also log as flattened params for searchability
    if "source_dataset" in feature_metadata:
        mlflow.log_params(
            {
                "dataset_uri": feature_metadata["source_dataset"]["source_uri"],
                "dataset_version_id": feature_metadata["source_dataset"]["version_id"],
                "dataset_last_modified": feature_metadata["source_dataset"]["last_modified"],
            }
        )

    # Log the pipeline as a single model
    signature = infer_signature(X_train, pipeline.predict(X_train))
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path=algorithm,
        registered_model_name="default-payment-" + algorithm.lower(),
        signature=signature,
        input_example=X_train.head(3),
    )

    logger.success(
        f"Pipeline model registered in MLflow. Run ID: {mlflow.active_run().info.run_id}"
    )


@app.command()
def run_training_pipeline(
    features_path: str = "s3://" + S3_BUCKET + "/gold/credit_card_default_features.parquet",
    target_col: str = "default_payment_next_month",
    experiment_name: str = "baseline-models",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Run the training pipeline.

    Args:
        features_path: Path to features in Gold layer
        target_col: Name of target column
        experiment_name: MLflow experiment name
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
    """
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE STARTED")
    logger.info("=" * 60)

    # Load features with metadata
    X_train, X_test, y_train, y_test, feature_metadata = load_features_with_metadata(
        features_path=features_path,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info(f"Feature version: {feature_metadata.get('feature_version')}")
    logger.info(
        f"Engineered features: {', '.join(feature_metadata.get('engineered_features', []))}"
    )

    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name}")

    # Train multiple models
    models = [logistic_regression_model(), random_forest_model()]

    for model in models:
        algorithm = model.__class__.__name__
        logger.info("-" * 60)
        logger.info(f"Training {algorithm}...")

        with mlflow.start_run(run_name=f"{algorithm}_{feature_metadata.get('feature_version')}"):
            # Create and train pipeline
            pipeline = create_pipeline(X_train, model)
            pipeline.fit(X_train, y_train)
            logger.success(f"{algorithm} training completed")

            # Evaluate
            metrics, cm, y_proba = evaluate_model(pipeline, X_test, y_test)
            logger.info(f"Metrics: {metrics}")

            # Log to MLflow
            log_model_run(pipeline, X_train, X_test, metrics, cm, algorithm, feature_metadata)

    logger.info("=" * 60)
    logger.success("TRAINING PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Models trained: {len(models)}")
    logger.info(f"Experiment: {experiment_name}")


if __name__ == "__main__":
    app()
