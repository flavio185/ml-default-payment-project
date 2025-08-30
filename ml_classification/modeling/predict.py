from pathlib import Path

from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
import typer

from ml_classification.config import PROCESSED_DATA_DIR, S3_BUCKET

app = typer.Typer()


@app.command()
def predict(
    features_path: str = "s3://" + S3_BUCKET + "/gold/credit_card_default_features.parquet",
    model_uri: str = "models:/default-payment-randomforestclassifier/14",
    predictions_path: Path = PROCESSED_DATA_DIR / "inference_results.parquet",
):
    """Generate predictions using the trained pipeline."""
    logger.info(f"Loading features from: {features_path}")
    X = pd.read_parquet(features_path, storage_options={"anon": False})
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
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(predictions_path, index=False)
    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
