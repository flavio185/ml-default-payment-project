import json
from pathlib import Path

from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def predict(
    model_uri: str = "models:/default-payment-randomforestclassifier/1",
    predictions_path: Path = Path("data/processed/inference_results.parquet"),
):
    """Generate predictions using a registered model version in MLflow 3."""
    model_name = model_uri.split("/")[1].split(":")[0]
    model_version = model_uri.split("/")[-1]
    logger.info(f"Loading model from: {model_uri}")
    pipeline = mlflow.sklearn.load_model(model_uri)

    # Fetch metadata attached to the model version
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(model_name, model_version)

    # Get associated metadata artifact (e.g. s3_metadata.json)
    artifacts = client.list_artifacts(mv.run_id, path="")
    if any(a.path == "s3_metadata.json" for a in artifacts):
        local_path = client.download_artifacts(mv.run_id, "s3_metadata.json")
        with open(local_path) as f:
            metadata = json.load(f)

        features_path = metadata["s3_uri"]
        version_id = metadata["version_id"]
        logger.info(f"Using dataset from {features_path} (version {version_id})")
    else:
        raise ValueError("No dataset metadata found in model artifacts")

    # Load features
    X = pd.read_parquet(features_path, storage_options={"anon": False})
    X = X.drop(columns=["default_payment_next_month"], errors="ignore")

    # Predict
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    results = X.copy()
    results["prediction"] = y_pred
    results["probability"] = y_proba

    # Save results
    logger.info(f"Saving inference results to: {predictions_path}")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(predictions_path, index=False)
    logger.success("Inference complete.")
