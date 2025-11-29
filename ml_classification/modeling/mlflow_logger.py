"""MLflow logging utilities.

This module handles all MLflow logging operations.
"""

from loguru import logger
from matplotlib import pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline


class MLflowExperimentLogger:
    """Handles MLflow logging for model training experiments."""

    def __init__(self, experiment_name: str):
        """Initialize MLflow experiment logger.

        Args:
            experiment_name: Name of MLflow experiment
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")

    def log_training_run(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        metrics: dict,
        confusion_matrix,
        feature_metadata: dict,
        run_name: str = None,
    ):
        """Log a complete training run to MLflow.

        Args:
            pipeline: Trained sklearn pipeline
            X_train: Training features
            X_test: Test features
            metrics: Evaluation metrics dictionary
            confusion_matrix: Confusion matrix plot
            feature_metadata: Feature metadata dictionary
            run_name: Optional name for the run
        """
        algorithm = pipeline.steps[-1][1].__class__.__name__

        with mlflow.start_run(run_name=run_name or algorithm):
            # Log basic parameters
            self._log_data_params(X_train, X_test)

            # Log algorithm
            mlflow.log_param("algorithm", algorithm)

            # Log feature information
            self._log_feature_params(feature_metadata)

            # Log all metrics
            mlflow.log_metrics(metrics)

            # Log confusion matrix
            self._log_confusion_matrix(confusion_matrix)

            # Log feature metadata as artifact
            mlflow.log_dict(feature_metadata, "feature_metadata.json")

            # Log dataset metadata if available
            self._log_dataset_metadata(feature_metadata)

            # Log the model
            self._log_model(pipeline, X_train, algorithm)

            run_id = mlflow.active_run().info.run_id
            logger.success(f"MLflow run logged successfully. Run ID: {run_id}")

    def _log_data_params(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Log data-related parameters."""
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])

    def _log_feature_params(self, feature_metadata: dict):
        """Log feature-related parameters."""
        mlflow.log_param("feature_version", feature_metadata.get("feature_version", "unknown"))
        mlflow.log_param("total_features", feature_metadata.get("total_columns"))
        mlflow.log_param(
            "engineered_features_count",
            len(feature_metadata.get("engineered_features", [])),
        )

        # Log engineered feature names
        engineered_features = feature_metadata.get("engineered_features", [])
        if engineered_features:
            mlflow.log_param("engineered_features", ",".join(engineered_features))

    def _log_confusion_matrix(self, cm):
        """Log confusion matrix plot."""
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

    def _log_dataset_metadata(self, feature_metadata: dict):
        """Log dataset metadata for lineage tracking."""
        if "source_dataset" in feature_metadata:
            source_dataset = feature_metadata["source_dataset"]
            mlflow.log_params(
                {
                    "dataset_uri": source_dataset.get("source_uri", "unknown"),
                    "dataset_version_id": source_dataset.get("version_id", "unknown"),
                    "dataset_last_modified": source_dataset.get("last_modified", "unknown"),
                }
            )

    def _log_model(self, pipeline: Pipeline, X_train: pd.DataFrame, algorithm: str):
        """Log the trained model to MLflow."""
        signature = infer_signature(X_train, pipeline.predict(X_train))

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path=algorithm,
            registered_model_name=f"default-payment-{algorithm.lower()}",
            signature=signature,
            input_example=X_train.head(3),
        )

        logger.info(f"Model registered: default-payment-{algorithm.lower()}")
