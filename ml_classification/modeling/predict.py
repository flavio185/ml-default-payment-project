from pathlib import Path

from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
import typer

from ml_classification.config import GOLD_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def predict(
    features_path: Path = GOLD_DATA_DIR / "credit_card_default_features.parquet",
    model_uri: str = "models:/default-payment-svc/1",
    predictions_path: Path = PROCESSED_DATA_DIR / "inference_results.parquet",
):
    """Generate predictions using the trained pipeline."""
    logger.info(f"Loading features from: {features_path}")
    X = pd.read_parquet(features_path)
    X = X.drop(columns=["default_payment_next_month"], errors="ignore")

    logger.info(f"Loading pipeline from: {model_uri}")
    pipeline = mlflow.sklearn.load_model(model_uri)

    logger.info("Generating predictions...")
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    results = X.copy()
    results["prediction"] = y_pred
    results["probability"] = y_proba

    logger.info(f"Saving inference results to: {predictions_path}")
    results.to_parquet(predictions_path, index=False)
    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
