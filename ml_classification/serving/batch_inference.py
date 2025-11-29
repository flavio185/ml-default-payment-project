"""Batch inference wrapper - uses the inference pipeline.

This is a wrapper around the inference pipeline for backward compatibility.
For new code, use ml_classification.pipelines.inference_pipeline directly.
"""

from loguru import logger
import typer

from ml_classification.pipelines.inference_pipeline import run_batch_inference

app = typer.Typer()


@app.command()
def batch_inference(
    model_uri: str = typer.Argument(
        ..., help="MLflow model URI (e.g. models:/default-payment/14)"
    ),
    source_table: str = typer.Argument(..., help="Input parquet/Delta/S3 path with raw data"),
    destination_table: str = typer.Argument(
        ..., help="Output parquet/Delta/S3 path for predictions"
    ),
):
    """
    Run batch inference on raw data using an MLflow model.

    This function wraps the inference pipeline, which:
    1. Loads raw data
    2. Applies the same feature engineering as training
    3. Generates predictions using the MLflow model
    4. Saves results with predictions and probabilities

    Note: This is a legacy wrapper. For new code, use:
    ml_classification.pipelines.inference_pipeline.run_batch_inference
    """
    logger.info("Running batch inference via inference pipeline...")

    # Delegate to the inference pipeline
    run_batch_inference(
        model_uri=model_uri, input_path=source_table, output_path=destination_table
    )

    logger.success("Batch inference complete via pipeline")


if __name__ == "__main__":
    app()
