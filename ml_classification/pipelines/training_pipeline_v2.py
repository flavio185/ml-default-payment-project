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

    # 3. Train multiple models, tracking whichever scores best on the
    # primary metric so it can be promoted to `champion` once all
    # candidates have been logged.
    models = [logistic_regression_model(), random_forest_model()]
    primary_metric = "roc_auc"
    best_run_id = None
    best_score = float("-inf")

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
        run_id = mlflow_logger.log_training_run(
            pipeline=trained_pipeline,
            X_train=X_train,
            X_test=X_test,
            metrics=metrics,
            confusion_matrix=cm,
            feature_metadata=feature_metadata,
            run_name=run_name,
        )

        score = metrics.get(primary_metric, float("-inf"))
        if score > best_score:
            best_score = score
            best_run_id = run_id

    # 4. Promote the best of today's candidates to `champion` only if it
    # actually beats the current champion — otherwise every run reassigns
    # champion to whichever candidate merely won this round, even when both
    # are worse than what's already deployed, and the promote DAG step ends
    # up opening a PR on every single run. (responsibility: mlflow_logger)
    champion_score = mlflow_logger.get_champion_score(primary_metric)
    promoted = False
    if not best_run_id:
        logger.warning("No candidate produced a valid score; skipping promotion")
    elif champion_score is None:
        logger.info("No existing champion — promoting best candidate unconditionally")
        mlflow_logger.promote_to_champion(best_run_id)
        promoted = True
    elif best_score > champion_score:
        logger.info(
            f"New best ({primary_metric}={best_score:.4f}) beats champion "
            f"({primary_metric}={champion_score:.4f})"
        )
        mlflow_logger.promote_to_champion(best_run_id)
        promoted = True
    else:
        logger.info(
            f"New best ({primary_metric}={best_score:.4f}) does not beat champion "
            f"({primary_metric}={champion_score:.4f}); keeping current champion"
        )

    logger.info("=" * 60)
    logger.success("TRAINING PIPELINE V2 COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Models trained: {len(models)}")
    logger.info(f"Experiment: {experiment_name}")
    if promoted:
        logger.info(f"New champion ({primary_metric}={best_score:.4f}): {best_run_id}")
    else:
        logger.info(
            f"Champion unchanged ({primary_metric}={champion_score:.4f}); "
            f"best candidate this run scored {best_score:.4f}"
        )


if __name__ == "__main__":
    app()
