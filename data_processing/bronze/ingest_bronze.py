from pathlib import Path

from loguru import logger
import typer
import pandas as pd
from ml_classification.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = RAW_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Downloading dataset...")
    # load the data
    df = pd.read_csv(
        "https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv",
        header=1,
        index_col=0,
    )

    df.to_csv(output_path)

    logger.success("Downloading dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
