from pathlib import Path
import pandas as pd
from loguru import logger
import typer

from ml_classification.config import SILVER_DATA_DIR, GOLD_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = SILVER_DATA_DIR / "credit_card_default.parquet",
    output_path: Path = GOLD_DATA_DIR / "credit_card_default_features.parquet",
):
    logger.info("Loading Silver dataset...")
    df = pd.read_parquet(input_path)

    # convert all columns to lowercase
    df.columns = df.columns.str.lower()

    # ---------------- Feature Engineering ----------------
    logger.info("Creating features...")

    # Age binning
    bins = [17, 25, 35, 50, 120]
    labels = ["18-25", "26-35", "36-50", "50+"]
    df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels)

    # Bill trend (difference between last and first bill amount)
    df["bill_trend"] = df["bill_amt6"] - df["bill_amt1"]

    # Pay ratio (total paid / total billed)
    total_pay = df[["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]].sum(axis=1)
    total_bill = df[["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"]].sum(axis=1)
    df["pay_ratio"] = total_pay / (total_bill.replace(0, 1))  # avoid division by zero

    # Utilization (last bill / credit limit)
    df["utilization"] = df["bill_amt6"] / df["limit_bal"]

    # ------------------------------------------------------
    logger.info(f"Saving Gold dataset to {output_path}...")
    df.to_parquet(output_path, index=False)
    logger.success("Gold dataset created successfully!")


if __name__ == "__main__":
    app()
