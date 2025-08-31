from datetime import datetime
import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
import typer

app = typer.Typer()

@app.command()
def batch_inference(
    model_uri: str = typer.Argument(..., help="MLflow model URI (e.g. models:/default-payment/14)"),
    source_table: str = typer.Argument(..., help="Input parquet/Delta/S3 path with features"),
    destination_table: str = typer.Argument(..., help="Output parquet/Delta/S3 path for predictions"),
):
    """
    Run batch inference on a table of features using an MLflow model.
    """

    logger.info(f"Loading source data from: {source_table}")
    if source_table.startswith("s3://") and source_table.endswith(".parquet"):
        X = pd.read_parquet(source_table, storage_options={"anon": False})
    else:
        X = pd.read_parquet(source_table)
    if "default_payment_next_month" in X.columns:
        X = X.drop(columns=["default_payment_next_month", 'ingestion_time'])  # drop labels if present
    for col in X.select_dtypes(include=["object", "category"]):
        X[col] = X[col].astype(str)
    if X.columns[0] in ["Unnamed: 0", "index"]:
        X = X.drop(X.columns[0], axis=1)
        
    logger.info(f"Loading model from MLflow: {model_uri}")
    pipeline = mlflow.sklearn.load_model(model_uri)

    logger.info("Running predictions...")
    y_proba = pipeline.predict_proba(X)[:, 1]
    if isinstance(y_proba, pd.DataFrame):
        y_proba = y_proba.values  # ensure numpy array
    if y_proba.ndim > 1 and y_proba.shape[1] > 1:
        y_pred = (y_proba[:, 1] >= 0.5).astype(int)
        y_proba = y_proba[:, 1]
    else:
        y_pred = (y_proba >= 0.5).astype(int)

    # Attach predictions
    results = X.copy()
    results["prediction"] = y_pred
    results["probability"] = y_proba

    results["inference_timestamp"] = datetime.utcnow().isoformat()

    logger.info(f"Saving predictions to: {destination_table}")
    if destination_table.startswith("s3://") or destination_table.endswith(".parquet"):
        results.to_parquet(destination_table, index=False, storage_options={"anon": False})
    else:
        # Could be Delta/Iceberg if you’re using lakehouse
        results.to_parquet(destination_table, index=False)

    logger.success("Batch inference complete ✅")


if __name__ == "__main__":
    app()
