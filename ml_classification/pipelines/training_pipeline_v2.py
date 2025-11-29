"""Training Pipeline V2 - Refactored with externalized responsibilities.

This pipeline orchestrates model training with clear separation of concerns:
- Data loading: data_loader.py
- Pipeline creation: pipeline_builder.py
- Training: trainer.py
- MLflow logging: mlflow_logger.py
"""

from loguru import logger
import typer

from ml_classification.config import S3_BUCKET
from ml_classification.modeling.data_loader import load_features
from ml_classification.modeling.mlflow_logger import MLflowExperimentLogger
from ml_classification.modeling.models import logistic_regression_model, random_forest_model
from ml_classification.modeling.pipeline_builder import create_sklearn_pipeline
from ml_classification.modeling.trainer import train_and_evaluate

app = typer.Typer()


@app.command()
def run_training_pipeline(
    features_path: str = "s3://" + S3_BUCKET + "/gold/credit_card_default_features.parquet",
    target_col: str = "default_payment_next_month",
    experiment_name: str = "baseline-models",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Run the training pipeline with externalized responsibilities.

    This pipeline is much simpler - it just orchestrates the components:
    1. Load data (data_loader)
    2. Create pipeline (pipeline_builder)
    3. Train and evaluate (trainer)
    4. Log to MLflow (mlflow_logger)

    Args:
        features_path: Path to features in Gold layer
        target_col: Name of target column
        experiment_name: MLflow experiment name
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
    """
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE V2 STARTED")
    logger.info("=" * 60)

    # 1. Load data (responsibility: data_loader)
    X_train, X_test, y_train, y_test, feature_metadata = load_features(
        features_path=features_path,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info(f"Feature version: {feature_metadata.get('feature_version')}")
    logger.info(
        f"Engineered features: {', '.join(feature_metadata.get('engineered_features', []))}"
    )

    # 2. Initialize MLflow logger (responsibility: mlflow_logger)
    mlflow_logger = MLflowExperimentLogger(experiment_name)

    # 3. Train multiple models
    models = [logistic_regression_model(), random_forest_model()]

    for model in models:
        algorithm = model.__class__.__name__
        logger.info("-" * 60)
        logger.info(f"Training {algorithm}...")

        # 3a. Create pipeline (responsibility: pipeline_builder)
        pipeline = create_sklearn_pipeline(X_train, model)

        # 3b. Train and evaluate (responsibility: trainer)
        trained_pipeline, metrics, cm, y_proba = train_and_evaluate(
            pipeline, X_train, y_train, X_test, y_test
        )

        # 3c. Log to MLflow (responsibility: mlflow_logger)
        run_name = f"{algorithm}_{feature_metadata.get('feature_version')}"
        mlflow_logger.log_training_run(
            pipeline=trained_pipeline,
            X_train=X_train,
            X_test=X_test,
            metrics=metrics,
            confusion_matrix=cm,
            feature_metadata=feature_metadata,
            run_name=run_name,
        )

    logger.info("=" * 60)
    logger.success("TRAINING PIPELINE V2 COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Models trained: {len(models)}")
    logger.info(f"Experiment: {experiment_name}")


if __name__ == "__main__":
    app()
