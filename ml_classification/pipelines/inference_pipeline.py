"""Inference Pipeline - Orchestrates batch inference with consistent feature engineering.

This pipeline ensures that the same feature engineering is applied during inference
as was used during training, preventing train-serve skew.
"""

from datetime import datetime, timezone

from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
import typer

from ml_classification.features.engineering import engineer_features

app = typer.Typer()


def prepare_inference_data(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    """Prepare data for inference by applying feature engineering.

    Args:
        df: Raw input DataFrame
        target_col: Optional target column name to drop if present

    Returns:
        DataFrame with engineered features ready for inference
    """
    logger.info("Preparing data for inference...")

    # Apply the same feature engineering as training
    df_features = engineer_features(df)

    # Drop target column if present (for evaluation scenarios)
    if target_col and target_col in df_features.columns:
        logger.info(f"Dropping target column: {target_col}")
        df_features = df_features.drop(columns=[target_col])

    # Drop metadata columns
    df_features = df_features.drop(columns=["ingestion_time"], errors="ignore")

    # Ensure categorical columns are strings (required for sklearn encoders)
    for col in df_features.select_dtypes(include=["object", "category"]):
        df_features[col] = df_features[col].astype(str)

    # Drop any index columns that may have been added
    if df_features.columns[0] in ["Unnamed: 0", "index"]:
        df_features = df_features.drop(df_features.columns[0], axis=1)

    logger.info(f"Data prepared: {len(df_features)} rows, {len(df_features.columns)} columns")
    return df_features


def load_model_from_mlflow(model_uri: str):
    """Load trained model from MLflow.

    Args:
        model_uri: MLflow model URI (e.g., 'models:/default-payment/14')

    Returns:
        Loaded sklearn pipeline
    """
    logger.info(f"Loading model from MLflow: {model_uri}")
    pipeline = mlflow.sklearn.load_model(model_uri)
    logger.success("Model loaded successfully")
    return pipeline


def generate_predictions(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate predictions using the loaded pipeline.

    Args:
        pipeline: Trained sklearn pipeline
        X: Features DataFrame

    Returns:
        Tuple of (input_features, predictions, probabilities)
    """
    logger.info("Generating predictions...")

    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    logger.success(f"Predictions generated for {len(X)} rows")
    logger.info(
        f"Prediction distribution: {sum(y_pred)} positive ({sum(y_pred) / len(y_pred) * 100:.1f}%)"
    )

    return X, y_pred, y_proba


def save_predictions(
    X: pd.DataFrame, predictions: pd.Series, probabilities: pd.Series, output_path: str
) -> None:
    """Save predictions to output location.

    Args:
        X: Input features
        predictions: Binary predictions
        probabilities: Prediction probabilities
        output_path: Path to save predictions (local or S3)
    """
    # Combine features with predictions
    results = X.copy()
    results["prediction"] = predictions
    results["probability"] = probabilities
    results["inference_timestamp"] = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Save to parquet
    logger.info(f"Saving predictions to: {output_path}")
    if output_path.startswith("s3://") or output_path.endswith(".parquet"):
        results.to_parquet(output_path, index=False, storage_options={"anon": False})
    else:
        results.to_parquet(output_path, index=False)

    logger.success(f"Predictions saved: {len(results)} rows")


@app.command()
def run_batch_inference(
    model_uri: str = typer.Argument(
        ..., help="MLflow model URI (e.g. models:/default-payment/14)"
    ),
    input_path: str = typer.Argument(..., help="Input data path (S3 or local)"),
    output_path: str = typer.Argument(..., help="Output path for predictions (S3 or local)"),
    target_col: str = typer.Option(
        "default_payment_next_month", help="Target column to drop if present"
    ),
):
    """Run batch inference pipeline.

    This pipeline:
    1. Loads raw data from input_path
    2. Applies the same feature engineering as training
    3. Loads the model from MLflow
    4. Generates predictions
    5. Saves results to output_path

    Args:
        model_uri: MLflow model URI
        input_path: Path to input data
        output_path: Path to save predictions
        target_col: Target column name to drop if present
    """
    logger.info("=" * 60)
    logger.info("INFERENCE PIPELINE STARTED")
    logger.info("=" * 60)
    logger.info(f"Model URI: {model_uri}")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")

    # Load input data
    logger.info("Loading input data...")
    if input_path.startswith("s3://") and input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path, storage_options={"anon": False})
    else:
        df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows")

    # Prepare data (apply feature engineering)
    X = prepare_inference_data(df, target_col=target_col)

    # Load model
    pipeline = load_model_from_mlflow(model_uri)

    # Generate predictions
    X, predictions, probabilities = generate_predictions(pipeline, X)

    # Save predictions
    save_predictions(X, predictions, probabilities, output_path)

    logger.info("=" * 60)
    logger.success("INFERENCE PIPELINE COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
