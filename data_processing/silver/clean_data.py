from pathlib import Path
import pandas as pd
import typer
from loguru import logger

from ml_classification.config import BRONZE_DATA_DIR, SILVER_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = BRONZE_DATA_DIR / "credit_card_default.parquet",
    output_path: Path = SILVER_DATA_DIR / "credit_card_default.parquet",
):
    """
    Day 3 – Silver Layer Cleaning
    Load Bronze dataset, apply minimal cleaning, and save to Silver.
    """

    logger.info(f"Loading Bronze dataset from: {input_path}")
    df = pd.read_parquet(input_path)

    # --- Cleaning steps ---
    logger.info("Renaming columns to snake_case...")
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    logger.info("Fixing invalid values in EDUCATION and MARRIAGE...")
    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})  # group invalid into "others"
    df["marriage"] = df["marriage"].replace({0: 3})  # group invalid into "others"

    logger.info("Ensuring correct data types...")
    categorical_cols = ["sex", "education", "marriage", "default_payment_next_month"]
    df[categorical_cols] = df[categorical_cols].astype("int")

    float_cols = [c for c in df.columns if "bill_amt" in c or "pay_amt" in c or c == "limit_bal"]
    df[float_cols] = df[float_cols].astype("float")

    # --- Save Silver ---
    SILVER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(f"Silver dataset saved at: {output_path}")


if __name__ == "__main__":
    app()
