"""Pipeline builder for creating sklearn pipelines.

This module handles creating ML pipelines with preprocessing and models.
"""

from loguru import logger
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_classification.features.preprocessing import build_preprocessor


def create_sklearn_pipeline(X_train: pd.DataFrame, model) -> Pipeline:
    """Create sklearn pipeline with preprocessing and model.

    Args:
        X_train: Training features for building preprocessor
        model: Sklearn model instance

    Returns:
        Pipeline with preprocessor and model
    """
    logger.info(f"Building pipeline with {model.__class__.__name__}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", model),
        ]
    )

    return pipeline


def get_pipeline_info(pipeline: Pipeline) -> dict:
    """Extract information about the pipeline.

    Args:
        pipeline: Trained sklearn pipeline

    Returns:
        Dictionary with pipeline information
    """
    info = {
        "steps": [step[0] for step in pipeline.steps],
        "classifier": pipeline.steps[-1][1].__class__.__name__,
        "n_features_in": getattr(pipeline, "n_features_in_", None),
    }

    # Get preprocessor info
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor:
        info["transformers"] = [name for name, _, _ in preprocessor.transformers]

    return info
