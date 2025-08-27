from pathlib import Path

from loguru import logger
import typer
import pandas as pd
from ml_classification.config import BRONZE_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # input_path: Path = RAW_DATA_DIR / "dataset.csv",
    dataset_source_url: str = "https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv",
    output_path: Path = BRONZE_DATA_DIR / "credit_card_default.parquet",
    # ----------------------------------------------
):
    
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Downloading dataset...")
    # load the data
    df = pd.read_csv(
        dataset_source_url,
        header=1,
        index_col=0,
    )
    df['ingestion_time'] = pd.Timestamp.now()
    df.to_parquet(output_path)

    logger.success(f"Dataset saved on {BRONZE_DATA_DIR}.")
    print(df.head())
    # -----------------------------------------


if __name__ == "__main__":
    app()
