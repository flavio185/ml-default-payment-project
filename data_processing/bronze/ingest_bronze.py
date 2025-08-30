from pathlib import Path

from loguru import logger
import typer
import pandas as pd
from ml_classification.config import S3_BUCKET

app = typer.Typer()


@app.command()
def main(
    input_path: str = "https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv",
    output_path: str = "s3://" + S3_BUCKET + "/bronze/credit_card_default.parquet"
):
    
    logger.info("Downloading dataset...")

    # load the data
    df = pd.read_csv(
        input_path,
        header=1,
        index_col=0,
    )
    df['ingestion_time'] = pd.Timestamp.now()
    df.to_parquet(output_path, storage_options={"anon": False})

    logger.success(f"Dataset saved on {output_path}.")
    print(df.head())
    # -----------------------------------------


if __name__ == "__main__":
    app()
