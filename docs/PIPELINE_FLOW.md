# Pipeline Data Flow

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYERS                                   │
└─────────────────────────────────────────────────────────────────────────┘

Raw Data (CSV/API)
       │
       ▼
┌──────────────┐
│   BRONZE     │  ← data_processing/bronze/ingest_bronze.py
│ (Raw parquet)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   SILVER     │  ← data_processing/silver/clean_data.py
│  (Cleaned)   │  ← data_processing/silver/validate_data.py
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    GOLD      │  ← ml_classification/pipelines/feature_pipeline.py
│  (Features)  │     • engineer_features()
│              │     • save feature metadata
│   + metadata │     • save preprocessing config
└──────┬───────┘
       │
       ├─────────────────────────────────────────────────┐
       │                                                 │
       ▼                                                 ▼
┌──────────────┐                                 ┌──────────────┐
│   TRAINING   │                                 │  INFERENCE   │
│              │                                 │              │
│ • Load Gold  │                                 │ • Load Raw   │
│ • Build      │                                 │ • Apply      │
│   Pipeline   │                                 │   Features   │
│ • Train      │                                 │ • Load Model │
│ • Evaluate   │                                 │ • Predict    │
│ • Log MLflow │                                 │ • Save       │
└──────┬───────┘                                 └──────────────┘
       │
       ▼
┌──────────────┐
│   MLFLOW     │
│ • Models     │
│ • Metrics    │
│ • Artifacts  │
└──────────────┘
```

## Pipeline Separation

### 1. Feature Pipeline (Independent)

```
Input:  s3://bucket/silver/credit_card_default.parquet
        │
        ▼
    ┌─────────────────────────────────┐
    │  Feature Engineering            │
    │  • create_age_bins()            │
    │  • create_bill_trend()          │
    │  • create_pay_ratio()           │
    │  • create_utilization()         │
    └────────────┬────────────────────┘
                 │
                 ▼
Output: s3://bucket/gold/credit_card_default_features.parquet
        s3://bucket/gold/credit_card_default_features_metadata.json
        s3://bucket/gold/credit_card_default_features_preprocessing_config.json

Schedule: Daily (or when new data arrives)
Duration: ~1-5 minutes
Can run: Independently
```

### 2. Training Pipeline (Uses Features)

```
Input:  s3://bucket/gold/credit_card_default_features.parquet
        + feature_metadata.json
        │
        ▼
    ┌─────────────────────────────────┐
    │  Load Features & Metadata       │
    │  • feature version              │
    │  • preprocessing config         │
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────┐
    │  Train Multiple Models          │
    │  • LogisticRegression           │
    │  • RandomForestClassifier       │
    │  • (add more models)            │
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────┐
    │  Evaluate & Log to MLflow       │
    │  • Metrics (accuracy, F1, etc)  │
    │  • Confusion matrix             │
    │  • Feature version tracking     │
    └────────────┬────────────────────┘
                 │
                 ▼
Output: MLflow registered models
        • models:/default-payment-logisticregression/v1
        • models:/default-payment-randomforestclassifier/v1

Schedule: Weekly/Monthly (or on-demand)
Duration: ~5-30 minutes (depending on data size)
Can run: After feature pipeline
```

### 3. Inference Pipeline (Applies Features)

```
Input:  s3://bucket/new_data/batch_20250121.parquet
        + MLflow model URI
        │
        ▼
    ┌─────────────────────────────────┐
    │  Apply Feature Engineering      │
    │  (Same as training!)            │
    │  • engineer_features()          │
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────┐
    │  Load Model from MLflow         │
    │  models:/default-payment/latest │
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────┐
    │  Generate Predictions           │
    │  • Binary predictions (0/1)     │
    │  • Probabilities (0.0-1.0)      │
    └────────────┬────────────────────┘
                 │
                 ▼
Output: s3://bucket/predictions/batch_20250121_predictions.parquet
        • Original features
        • prediction column
        • probability column
        • inference_timestamp

Schedule: Hourly/Daily (or on-demand)
Duration: ~1-10 minutes
Can run: Independently (just needs model URI)
```

## Key Insight: Feature Reuse

```
┌─────────────────────────────────────────────────────────────┐
│           features/engineering.py (Single Source)           │
│                                                             │
│  def engineer_features(df):                                 │
│      df = create_age_bins(df)                               │
│      df = create_bill_trend(df)                             │
│      df = create_pay_ratio(df)                              │
│      df = create_utilization(df)                            │
│      return df                                               │
└──────────────────┬──────────────────────┬───────────────────┘
                   │                      │
                   │                      │
         ┌─────────▼──────────┐  ┌────────▼──────────┐
         │  Feature Pipeline  │  │ Inference Pipeline│
         │                    │  │                   │
         │  Gold = engineer_  │  │ X = engineer_     │
         │    features(Silver)│  │   features(Raw)   │
         └────────────────────┘  └───────────────────┘

         ✅ Same code             ✅ Same code
         ✅ Same logic            ✅ Same logic
         ✅ Same result           ✅ Same result
```

**Result**: NO TRAIN-SERVE SKEW! 🎉

## Command Quick Reference

### Run Individual Pipelines

```bash
# 1. Feature Pipeline
python make.py feature-pipeline

# 2. Training Pipeline
python make.py training-pipeline

# 3. Inference Pipeline
python make.py inference-pipeline \
  models:/default-payment-randomforestclassifier/1 \
  s3://bucket/input.parquet \
  s3://bucket/output.parquet
```

### Run Full Pipeline

```bash
# Bronze → Silver → Validate → Gold → Train
python make.py full-pipeline
```

### Legacy Commands (Still Work)

```bash
python make.py gold      # → Use feature-pipeline instead
python make.py train     # → Use training-pipeline instead
python make.py pipeline  # → Use full-pipeline instead
```

## Typical Production Schedule

```bash
# Crontab example

# Daily: Ingest and create features
0 2 * * * cd /project && python make.py bronze
0 3 * * * cd /project && python make.py silver
0 4 * * * cd /project && python make.py feature-pipeline

# Weekly: Retrain models (Sunday 5 AM)
0 5 * * 0 cd /project && python make.py training-pipeline

# Hourly: Batch inference
0 * * * * cd /project && python make.py inference-pipeline \
  models:/default-payment/latest \
  s3://bucket/new_data.parquet \
  s3://bucket/predictions/$(date +\%Y\%m\%d_\%H).parquet
```

## Benefits at Each Stage

### Bronze Layer
✓ Raw data preserved
✓ Can reprocess from source
✓ Historical record

### Silver Layer
✓ Cleaned and validated
✓ Ready for analysis
✓ Multiple downstream uses

### Gold Layer (Features)
✓ **Versioned features**
✓ **Reusable across models**
✓ **Decoupled from training**
✓ **Preprocessing config saved**

### Training
✓ **Fast iteration** (features pre-computed)
✓ **Multiple models** in one run
✓ **Feature version tracking**
✓ **Full reproducibility**

### Inference
✓ **No train-serve skew**
✓ **Consistent transformations**
✓ **Scalable batch scoring**
✓ **Easy model updates**

## Monitoring Points

```
┌──────────────┐
│ Bronze       │ → Monitor: Data arrival, schema changes
└──────────────┘

┌──────────────┐
│ Silver       │ → Monitor: Data quality, validation failures
└──────────────┘

┌──────────────┐
│ Gold         │ → Monitor: Feature distributions, null rates
└──────────────┘

┌──────────────┐
│ Training     │ → Monitor: Model metrics, training time
└──────────────┘

┌──────────────┐
│ Inference    │ → Monitor: Prediction distribution, latency
└──────────────┘
```

## Next Steps

1. **Test the pipelines**: Run each pipeline separately
2. **Verify outputs**: Check S3 for generated files and metadata
3. **Update schedules**: Adjust cron jobs for your needs
4. **Add monitoring**: Track pipeline health and data quality
5. **Extend features**: Add new transformations to `engineering.py`
