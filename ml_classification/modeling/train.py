from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
import typer
from loguru import logger
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from ml_classification.config import GOLD_DATA_DIR
from ml_classification.modeling.data import build_preprocessor, load_data
from ml_classification.modeling.eval import evaluate_model


app = typer.Typer()


def train_logistic_regression(X_train, y_train):
    logger.info("Training Logistic Regression baseline...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def create_pipeline(X_train):
    model_params = {
        "C": 1.0,
        "max_iter": 100,
        "solver": "liblinear",
    }
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", LogisticRegression(**model_params)),
        ]
    )
    return pipeline



def log_experiment(pipeline, X_train, X_test, metrics, cm, experiment_name):

    mlflow.set_experiment(experiment_name)
    
    # Params
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])

    # Log all metrics
    mlflow.log_metrics(metrics)
    
    # Log confusion matrix plot
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    
    # Log the pipeline as a single model
    signature = infer_signature(X_train, pipeline.predict(X_train))
    mlflow.sklearn.log_model(
        pipeline,
        name="default-payment-class-logreg",
        registered_model_name=experiment_name,
        signature=signature,
        input_example=X_train.head(3)
    )
    logger.success(f"Pipeline model registered in MLflow. Run ID: {mlflow.active_run().info.run_id}")

@app.command()
def main(
    experiment_name: str = "baseline-logreg"
):
    data_path = GOLD_DATA_DIR / "credit_card_default_features.parquet"
    X_train, X_test, y_train, y_test = load_data(data_path)
    pipeline = create_pipeline(X_train)

    pipeline.fit(X_train, y_train)

    metrics, cm, y_proba = evaluate_model(pipeline, X_test, y_test)
    log_experiment(
        pipeline, X_train, X_test, metrics, cm, experiment_name
    )

if __name__== "__main__":
    app()