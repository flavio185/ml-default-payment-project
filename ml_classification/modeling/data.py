import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from loguru import logger


def build_preprocessor(X):
    categorical_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype == "category"]
    numerical_cols = [c for c in X.columns if X[c].dtype in ["int64", "float64"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )
    return preprocessor


def load_data(data_path) -> pd.DataFrame:
    target_col = "default_payment_next_month"
    logger.info(f"Loading dataset from: {data_path}")
    df = pd.read_parquet(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
