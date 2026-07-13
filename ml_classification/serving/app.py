from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from loguru import logger
import mlflow
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Online MLflow Inference API")


# -----------------------------
# Input data schemas
# -----------------------------
class FeatureRow(BaseModel):
    features: Dict[str, Any]


class FeatureBatch(BaseModel):
    data: List[FeatureRow]


class FeastPredictRequest(BaseModel):
    customer_ids: List[int]


# -----------------------------
# Load MLflow model (once)
# -----------------------------
MODEL_URI = "models:/default-payment-logisticregression/1"  # adjust as needed
logger.info(f"Loading model from MLflow: {MODEL_URI}")
pipeline = mlflow.sklearn.load_model(MODEL_URI)
signature = mlflow.models.get_model_info(MODEL_URI).signature
EXPECTED_COLS = [c.name for c in signature.inputs]


# -----------------------------
# Online inference endpoint (backward compatible — features in request)
# -----------------------------
@app.post("/predict")
def predict(batch: FeatureBatch):
    if not batch.data:
        raise HTTPException(status_code=400, detail="Empty input data")

    # Convert input to DataFrame
    X = pd.DataFrame([row.features for row in batch.data])

    # Ensure categorical columns are strings
    for col in X.select_dtypes(include=["object", "category"]):
        X[col] = X[col].astype(str)

    # Align columns to model signature
    missing_cols = set(EXPECTED_COLS) - set(X.columns)
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing expected columns: {missing_cols}")
    X = X[EXPECTED_COLS]

    # Run predictions
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Attach results
    results = X.copy()
    results["prediction"] = y_pred
    results["probability"] = y_proba
    results["inference_timestamp"] = datetime.utcnow().isoformat()

    return results.to_dict(orient="records")


# -----------------------------
# Feast-based inference endpoint — features looked up from online store
# -----------------------------
@app.post("/predict/feast")
def predict_feast(request: FeastPredictRequest):
    if not request.customer_ids:
        raise HTTPException(status_code=400, detail="Empty customer_ids list")

    try:
        from ml_classification.features.feast_utils import get_online_features
    except ImportError:
        raise HTTPException(status_code=501, detail="Feast not available")

    # Recupera features do online store
    entity_rows = [{"customer_id": cid} for cid in request.customer_ids]
    X = get_online_features(entity_rows)

    # Remove entity key column
    X = X.drop(columns=["customer_id"], errors="ignore")

    # Ensure categorical columns are strings
    for col in X.select_dtypes(include=["object", "category"]):
        X[col] = X[col].astype(str)

    # Align columns to model signature
    missing_cols = set(EXPECTED_COLS) - set(X.columns)
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing expected columns from Feast: {missing_cols}",
        )
    X = X[EXPECTED_COLS]

    # Run predictions
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    results = []
    for i, cid in enumerate(request.customer_ids):
        results.append(
            {
                "customer_id": cid,
                "prediction": int(y_pred[i]),
                "probability": float(y_proba[i]),
                "inference_timestamp": datetime.utcnow().isoformat(),
            }
        )

    return results


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "model_uri": MODEL_URI}
