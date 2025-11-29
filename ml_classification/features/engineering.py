"""Feature engineering functions for credit card default prediction.

This module contains all feature engineering logic that transforms
raw data into features used for model training and inference.
"""

from loguru import logger
import pandas as pd


def create_age_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Create age bin categorical feature.

    Args:
        df: DataFrame with 'age' column

    Returns:
        DataFrame with added 'age_bin' column
    """
    bins = [17, 25, 35, 50, 120]
    labels = ["18_25", "26_35", "36_50", "50_plus"]
    df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels)
    return df


def create_bill_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Create bill trend feature (difference between last and first bill amount).

    Args:
        df: DataFrame with 'bill_amt1' and 'bill_amt6' columns

    Returns:
        DataFrame with added 'bill_trend' column
    """
    df["bill_trend"] = df["bill_amt6"] - df["bill_amt1"]
    return df


def create_pay_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Create pay ratio feature (total paid / total billed).

    Args:
        df: DataFrame with payment and bill amount columns

    Returns:
        DataFrame with added 'pay_ratio' column
    """
    total_pay = df[["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]].sum(
        axis=1
    )
    total_bill = df[
        ["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"]
    ].sum(axis=1)
    # Avoid division by zero
    df["pay_ratio"] = total_pay / total_bill.replace(0, 1)
    return df


def create_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """Create utilization feature (last bill / credit limit).

    Args:
        df: DataFrame with 'bill_amt6' and 'limit_bal' columns

    Returns:
        DataFrame with added 'utilization' column
    """
    df["utilization"] = df["bill_amt6"] / df["limit_bal"]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations.

    This is the main function that orchestrates all feature engineering steps.
    It should be used consistently in both training and inference pipelines
    to prevent train-serve skew.

    Args:
        df: DataFrame with raw features from Silver layer

    Returns:
        DataFrame with engineered features ready for Gold layer
    """
    logger.info("Starting feature engineering...")

    # Ensure all columns are lowercase
    df.columns = df.columns.str.lower()

    # Apply feature engineering functions
    df = create_age_bins(df)
    df = create_bill_trend(df)
    df = create_pay_ratio(df)
    df = create_utilization(df)

    logger.info("Feature engineering completed.")
    return df


def get_feature_names() -> list[str]:
    """Get list of all engineered feature names.

    Returns:
        List of feature names created by engineer_features()
    """
    return ["age_bin", "bill_trend", "pay_ratio", "utilization"]
