from pathlib import Path

from loguru import logger
import mlflow
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import typer
import pandas as pd


from ml_classification.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # labels_test_path: Path = PROCESSED_DATA_DIR / "labels_test.csv",
    # features_train_path: Path = PROCESSED_DATA_DIR / "features_train.csv",
    # labels_train_path: Path = PROCESSED_DATA_DIR / "labels_train.csv",    
    # model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # Enable complete experiment tracking with one line
    mlflow.sklearn.autolog(disable=True)
    logger.info("Training some model...")
    from sklearn.ensemble import RandomForestClassifier
    params = {'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 200}
    df = pd.read_csv(features_path)
    y = df.default
    X = df.drop("default", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(**params)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {acc}")

    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=rf,
        name="sklearn-model",
        input_example=X_train,
        registered_model_name="rf-churn-class-model",
    )
    for i in tqdm(range(10), total=10):
        if i == 9:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
