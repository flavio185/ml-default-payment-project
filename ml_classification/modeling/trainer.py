"""Model training utilities.

This module handles the actual training and evaluation of models.
"""

from loguru import logger
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_classification.modeling.eval import evaluate_model


def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a sklearn pipeline.

    Args:
        pipeline: Sklearn pipeline to train
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained pipeline
    """
    model_name = pipeline.steps[-1][1].__class__.__name__
    logger.info(f"Training {model_name}...")

    pipeline.fit(X_train, y_train)

    logger.success(f"{model_name} training completed")
    return pipeline


def train_and_evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict, object, object]:
    """Train and evaluate a model pipeline.

    Args:
        pipeline: Sklearn pipeline to train
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Tuple of (trained_pipeline, metrics, confusion_matrix, y_proba)
    """
    # Train
    trained_pipeline = train_model(pipeline, X_train, y_train)

    # Evaluate
    model_name = pipeline.steps[-1][1].__class__.__name__
    logger.info(f"Evaluating {model_name}...")

    metrics, cm, y_proba = evaluate_model(trained_pipeline, X_test, y_test)
    logger.info(f"Metrics: {metrics}")

    return trained_pipeline, metrics, cm, y_proba
