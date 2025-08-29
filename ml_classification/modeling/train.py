from loguru import logger
from matplotlib import pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from models import logistic_regression_model, random_forest_model
from sklearn.pipeline import Pipeline
import typer

from ml_classification.config import REPORTS_DIR
from ml_classification.modeling.data import build_preprocessor
from ml_classification.modeling.data import main as load_data
from ml_classification.modeling.eval import evaluate_model

app = typer.Typer()


def create_pipeline(X_train, model):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", model),
        ]
    )
    return pipeline


def log_model_run(pipeline, X_train, X_test, metrics, cm, algorithm):
    # Params
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])

    # Log all metrics
    mlflow.log_metrics(metrics)

    # Log confusion matrix plot
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    # log dataset URI and version as JSON

    import json

    metadata_path = REPORTS_DIR / "s3_metadata.json"
    with open(metadata_path, "r") as f:
        metadata_file = json.load(f)
    mlflow.log_param("dataset_uri", metadata_file["s3_uri"])
    mlflow.log_param("dataset_version", metadata_file["version_id"])
    mlflow.log_param("dataset_last_modified", metadata_file["split_strategy"])

    # Log the pipeline as a single model
    signature = infer_signature(X_train, pipeline.predict(X_train))
    mlflow.sklearn.log_model(
        pipeline,
        name=algorithm,
        registered_model_name="default-payment-" + algorithm.lower(),
        signature=signature,
        input_example=X_train.head(3),
    )
    logger.success(
        f"Pipeline model registered in MLflow. Run ID: {mlflow.active_run().info.run_id}"
    )


@app.command()
def main(experiment_name: str = "baseline-logreg"):
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment(experiment_name)
    for model in [logistic_regression_model(), random_forest_model()]:
        with mlflow.start_run():
            algorithm = model.__class__.__name__
            pipeline = create_pipeline(X_train, model)

            pipeline.fit(X_train, y_train)

            metrics, cm, y_proba = evaluate_model(pipeline, X_test, y_test)
            log_model_run(pipeline, X_train, X_test, metrics, cm, algorithm)


if __name__ == "__main__":
    app()
