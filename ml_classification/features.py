from pathlib import Path

from loguru import logger
import typer
import pandas as pd

from ml_classification.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    
    logger.info("Generating features from dataset...")
    raw_data = input_path
    df = pd.read_csv(raw_data)
    # rename column
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    # remove ID column
    df.drop("ID", axis=1, inplace=True)

    processed_data = output_path
    df.to_csv(processed_data, index=False)

    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
