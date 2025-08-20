from pathlib import Path
from datetime import datetime

import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np

from ml_classification.config import GOLD_DATA_DIR

app = typer.Typer()

def precision_recall_at_k(y_true, y_scores, k=0.1):
    threshold = np.quantile(y_scores, 1 - k)
    y_pred = (y_scores >= threshold).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return precision, recall

@app.command()
def main(
    input_path: Path = GOLD_DATA_DIR / "credit_card_default_features.parquet",
    experiment_name: str = "baseline-logreg"
):


    logger.info(f"Loading Gold dataset from: {input_path}")
    df = pd.read_parquet(input_path)
    if df.empty:
        logger.error("Input DataFrame is empty. Exiting training.")
        raise typer.Exit(code=1)

    target_col = "default_payment_next_month"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Automatically encode categorical features (object or category dtype)
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        logger.info(f"Encoding categorical columns: {categorical_cols}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train Logistic Regression
    logger.info("Training Logistic Regression baseline...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    precision_at_10, recall_at_10 = precision_recall_at_k(y_test, y_proba, k=0.1)

    logger.info(f"ROC-AUC: {roc_auc:.3f}")
    logger.info(f"Precision@10%: {precision_at_10:.3f}")
    logger.info(f"Recall@10%: {recall_at_10:.3f}")

    # MLflow experiment tracking
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="baseline-logreg") as run:
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision_at_10", precision_at_10)
        mlflow.log_metric("recall_at_10", recall_at_10)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            name="baseline_logreg",
            signature=signature,
            input_example=X_train.head(3)
        )
        logger.success(f"Baseline Logistic Regression model registrado no MLflow. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    app()
