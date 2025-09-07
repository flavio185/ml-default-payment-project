from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from loguru import logger
import mlflow
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Online MLflow Inference API")


# -----------------------------
# Input data schema
# -----------------------------
class FeatureRow(BaseModel):
    features: Dict[str, Any]


class FeatureBatch(BaseModel):
    data: List[FeatureRow]


# -----------------------------
# Load MLflow model (once)
# -----------------------------
MODEL_URI = "models:/default-payment-logisticregression/1"  # adjust as needed
logger.info(f"Loading model from MLflow: {MODEL_URI}")
pipeline = mlflow.sklearn.load_model(MODEL_URI)
signature = mlflow.models.get_model_info(MODEL_URI).signature
EXPECTED_COLS = [c.name for c in signature.inputs]


# -----------------------------
# Online inference endpoint
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
# Optional health check
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "model_uri": MODEL_URI}
