"""Drift Check - Compares newly-created Gold features against the champion
model's training data using the Population Stability Index (PSI).

Standalone batch job for now (not wired into the Argo DAG yet — run manually
or on a schedule after feature_pipeline.py, per feature_pipeline.py's own
docstring: "can run independently and on its own schedule"). Intended to
become a `drift-check` DAG step after `features` once there's a clear answer
for where results should surface beyond a local report (MLflow metrics,
Prometheus, etc.) -- deliberately out of scope for this first pass.

Baseline resolution: feature_pipeline.py now captures the S3 version_id of
each Gold parquet it writes (`gold_dataset` in the saved *_metadata.json),
so a champion trained on an older run can still be compared against its
*exact* Gold snapshot even though the Gold file at the fixed S3 key has
since been overwritten by later runs. Champion runs logged before that field
existed fall back to the nearest S3 object version at or before the run's
start time.
"""

from datetime import datetime, timezone
import io
import json

import boto3
from loguru import logger
import mlflow.artifacts
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import typer

from ml_classification.config import DRIFT_REPORTS_DIR, S3_BUCKET

app = typer.Typer()

# Columns that aren't feature signal -- an ID and the label -- excluded from PSI.
EXCLUDE_COLUMNS = {"customer_id", "default_payment_next_month", "ingestion_time"}

PSI_WARNING_THRESHOLD = 0.1
PSI_CRITICAL_THRESHOLD = 0.2


def compute_psi(baseline: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index between two samples of the same feature.

    Standard thresholds: < 0.1 no significant shift, 0.1-0.2 moderate
    (worth watching), > 0.2 significant drift.
    """
    baseline = baseline.dropna()
    current = current.dropna()

    if pd.api.types.is_numeric_dtype(baseline):
        edges = np.unique(baseline.quantile(np.linspace(0, 1, buckets + 1)).to_numpy())
        if len(edges) < 3:
            # Near-constant baseline (e.g. a flag column) -- quantile binning
            # degenerates to one edge; fall back to a single split point.
            edges = np.array([-np.inf, baseline.median(), np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        bin_edges = edges.tolist()
        baseline_counts = pd.cut(baseline, bin_edges).value_counts(sort=False)
        current_counts = pd.cut(current, bin_edges).value_counts(sort=False)
    else:
        categories = sorted(set(baseline.unique()) | set(current.unique()), key=str)
        baseline_counts = baseline.value_counts().reindex(categories, fill_value=0)
        current_counts = current.value_counts().reindex(categories, fill_value=0)

    # Clip so an empty bin never produces a log(0)/divide-by-zero.
    baseline_pct = (baseline_counts / len(baseline)).clip(lower=1e-4)
    current_pct = (current_counts / len(current)).clip(lower=1e-4)

    return float(((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)).sum())


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    bucket = uri.split("/")[2]
    key = "/".join(uri.split("/")[3:])
    return bucket, key


def resolve_champion_gold_uri(model_name: str) -> tuple[str, str, str]:
    """Return (gold_s3_uri, version_id, run_id) for the Gold snapshot the
    current champion was trained on."""
    client = MlflowClient()
    champion_version = client.get_model_version_by_alias(model_name, "champion")
    run_id = champion_version.run_id
    assert run_id is not None, f"champion version of {model_name} has no run_id"

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="feature_metadata.json"
    )
    with open(local_path) as f:
        feature_metadata = json.load(f)

    gold_uri = f"s3://{S3_BUCKET}/gold/credit_card_default_features.parquet"

    gold_dataset = feature_metadata.get("gold_dataset")
    if gold_dataset:
        return gold_uri, gold_dataset["version_id"], run_id

    # Fallback for champion runs logged before feature_pipeline.py captured
    # gold_dataset: find the S3 version closest to (at or before) run start.
    run = client.get_run(run_id)
    run_time_ms = run.info.start_time
    bucket, key = _parse_s3_uri(gold_uri)
    s3 = boto3.client("s3")
    versions = s3.list_object_versions(Bucket=bucket, Prefix=key).get("Versions", [])
    versions_before = [v for v in versions if v["LastModified"].timestamp() * 1000 <= run_time_ms]
    if not versions_before:
        raise ValueError(
            f"No Gold object version found at/before champion run {run_id}'s start time, "
            f"and its feature_metadata.json has no gold_dataset field to pin an exact one."
        )
    nearest = max(versions_before, key=lambda v: v["LastModified"])
    logger.warning(
        f"Champion run {run_id} predates gold_dataset tracking; using nearest S3 "
        f"version by timestamp instead: {nearest['VersionId']}"
    )
    return gold_uri, nearest["VersionId"], run_id


def read_parquet_version(uri: str, version_id: str) -> pd.DataFrame:
    """Read a specific S3 object version directly via boto3 -- avoids relying
    on fsspec/s3fs version-handling, which varies by version/config."""
    bucket, key = _parse_s3_uri(uri)
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key, VersionId=version_id)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


@app.command()
def run_drift_check(
    model_name: str = "default-payment-predictor",
    current_gold_path: str = "s3://" + S3_BUCKET + "/gold/credit_card_default_features.parquet",
    psi_warning_threshold: float = PSI_WARNING_THRESHOLD,
    psi_critical_threshold: float = PSI_CRITICAL_THRESHOLD,
):
    """Compare the latest Gold features against the champion's training data (PSI).

    Args:
        model_name: MLflow registered model name to resolve the champion alias from.
        current_gold_path: Gold parquet to check (defaults to the latest write).
        psi_warning_threshold: PSI at/above this is flagged WARNING.
        psi_critical_threshold: PSI at/above this is flagged CRITICAL.
    """
    logger.info("=" * 60)
    logger.info("DRIFT CHECK STARTED")
    logger.info("=" * 60)

    baseline_uri, baseline_version, champion_run_id = resolve_champion_gold_uri(model_name)
    logger.info(f"Baseline: {baseline_uri} (version {baseline_version}, run {champion_run_id})")
    baseline_df = read_parquet_version(baseline_uri, baseline_version)
    logger.info(f"Loaded {len(baseline_df)} baseline rows")

    logger.info(f"Current: {current_gold_path} (latest)")
    current_df = pd.read_parquet(current_gold_path, storage_options={"anon": False})
    logger.info(f"Loaded {len(current_df)} current rows")

    columns = [
        c for c in current_df.columns if c in baseline_df.columns and c not in EXCLUDE_COLUMNS
    ]

    results = {}
    for col in columns:
        psi = compute_psi(baseline_df[col], current_df[col])
        if psi >= psi_critical_threshold:
            status = "CRITICAL"
        elif psi >= psi_warning_threshold:
            status = "WARNING"
        else:
            status = "OK"
        results[col] = {"psi": round(psi, 4), "status": status}
        logger.info(f"  {col:20s} PSI={psi:.4f}  [{status}]")

    n_critical = sum(1 for r in results.values() if r["status"] == "CRITICAL")
    n_warning = sum(1 for r in results.values() if r["status"] == "WARNING")
    overall_status = "CRITICAL" if n_critical else "WARNING" if n_warning else "OK"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "baseline": {
            "uri": baseline_uri,
            "version_id": baseline_version,
            "champion_run_id": champion_run_id,
        },
        "current": {"uri": current_gold_path},
        "thresholds": {"warning": psi_warning_threshold, "critical": psi_critical_threshold},
        "overall_status": overall_status,
        "features": results,
    }

    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = DRIFT_REPORTS_DIR / f"drift_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    if overall_status == "CRITICAL":
        logger.error(
            f"DRIFT CHECK COMPLETED: {overall_status} ({n_critical} feature(s) drifted significantly)"
        )
    elif overall_status == "WARNING":
        logger.warning(
            f"DRIFT CHECK COMPLETED: {overall_status} ({n_warning} feature(s) show moderate drift)"
        )
    else:
        logger.success(f"DRIFT CHECK COMPLETED: {overall_status}")
    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    app()
