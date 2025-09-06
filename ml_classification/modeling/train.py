from loguru import logger
from matplotlib import pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from sklearn.pipeline import Pipeline
import typer

from ml_classification.config import S3_BUCKET
from ml_classification.modeling.data import build_preprocessor, load_data
from ml_classification.modeling.eval import evaluate_model
from ml_classification.modeling.models import logistic_regression_model, random_forest_model

app = typer.Typer()


def create_pipeline(X_train, model):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", model),
        ]
    )
    return pipeline


def log_model_run(pipeline, X_train, X_test, metrics, cm, algorithm, metadata):
    # Params
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])

    # Log all metrics
    mlflow.log_metrics(metrics)

    # Log confusion matrix plot
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    mlflow.log_dict(metadata, "ds_metadata.json")  # saved as artifact
    mlflow.log_params(
        {  # also flatten for searchability
            "dataset_uri": metadata["uri"],
            "dataset_version": metadata["version_id"],
            "dataset_last_modified": metadata["last_modified"],
        }
    )
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
    data_path = "s3://" + S3_BUCKET + "/gold/credit_card_default_features.parquet"
    target_col = "default_payment_next_month"
    X_train, X_test, y_train, y_test, metadata = load_data(
        data_path=data_path, target_col=target_col
    )

    mlflow.set_tracking_uri("http://127.0.0.1")
    mlflow.set_experiment(experiment_name)
    for model in [logistic_regression_model(), random_forest_model()]:
        with mlflow.start_run():
            algorithm = model.__class__.__name__
            pipeline = create_pipeline(X_train, model)

            pipeline.fit(X_train, y_train)

            metrics, cm, y_proba = evaluate_model(pipeline, X_test, y_test)
            log_model_run(pipeline, X_train, X_test, metrics, cm, algorithm, metadata)


if __name__ == "__main__":
    app()
