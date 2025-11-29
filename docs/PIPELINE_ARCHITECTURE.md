# Pipeline Architecture

This document describes the refactored ML pipeline architecture that separates concerns between feature engineering, model training, and inference.

## Overview

The project is now organized into three independent pipelines:

1. **Feature Pipeline**: Transforms raw data into features
2. **Training Pipeline**: Trains models using pre-computed features
3. **Inference Pipeline**: Applies models to new data with consistent feature engineering

## Directory Structure

```
ml_classification/
├── features/
│   ├── engineering.py      # Feature engineering functions
│   └── preprocessing.py    # Preprocessing utilities
├── pipelines/
│   ├── feature_pipeline.py    # Feature engineering orchestration
│   ├── training_pipeline.py   # Model training orchestration
│   └── inference_pipeline.py  # Batch inference orchestration
├── modeling/
│   ├── models.py          # Model definitions
│   ├── eval.py            # Evaluation metrics
│   ├── train.py           # Training logic (legacy)
│   └── data.py            # Data loading utilities
└── serving/
    ├── app.py             # FastAPI online inference
    └── batch_inference.py # Batch inference wrapper
```

## Pipeline Details

### 1. Feature Pipeline

**Purpose**: Transform Silver layer data into Gold layer features with versioning.

**Input**: Silver layer parquet (cleaned data)

**Output**:
- Gold layer parquet (features)
- Feature metadata JSON
- Preprocessing configuration JSON

**Key Functions**:
- `engineer_features()`: Applies all feature transformations
- Age binning, bill trends, pay ratios, utilization metrics

**Run Command**:
```bash
python make.py feature-pipeline
```

**Direct Command**:
```bash
uv run python ml_classification/pipelines/feature_pipeline.py
```

**Benefits**:
- Can run on its own schedule (e.g., daily)
- Creates versioned feature sets
- Saves preprocessing configuration for training
- No coupling to model training

### 2. Training Pipeline

**Purpose**: Train and evaluate models using pre-computed features.

**Input**:
- Gold layer features
- Feature metadata

**Output**:
- Trained model artifacts in MLflow
- Evaluation metrics
- Model performance visualizations

**Key Functions**:
- `load_features_with_metadata()`: Loads features with version tracking
- `create_pipeline()`: Builds sklearn pipeline with preprocessing
- `log_model_run()`: Logs everything to MLflow

**Run Command**:
```bash
python make.py training-pipeline
```

**Direct Command**:
```bash
uv run python ml_classification/pipelines/training_pipeline.py
```

**Benefits**:
- Decoupled from feature engineering
- Can retrain without recomputing features
- Tracks feature versions with models
- Multiple models trained in single run

### 3. Inference Pipeline

**Purpose**: Apply models to new data with consistent feature engineering.

**Input**:
- Raw data (similar to Silver layer)
- MLflow model URI

**Output**:
- Predictions with probabilities
- Inference timestamps

**Key Functions**:
- `prepare_inference_data()`: Applies same feature engineering as training
- `generate_predictions()`: Uses MLflow model for predictions
- `save_predictions()`: Saves results with metadata

**Run Command**:
```bash
python make.py inference-pipeline <model_uri> <input_path> <output_path>
```

**Direct Command**:
```bash
uv run python ml_classification/pipelines/inference_pipeline.py \
  models:/default-payment-logisticregression/1 \
  s3://bucket/data.parquet \
  s3://bucket/predictions.parquet
```

**Benefits**:
- No train-serve skew (uses same feature code)
- Applies feature engineering automatically
- Works with any MLflow registered model
- Suitable for batch scoring jobs

## Full Pipeline

Run all pipelines in sequence:

```bash
python make.py full-pipeline
```

This executes:
1. Bronze ingestion
2. Silver cleaning
3. Silver validation
4. Feature pipeline (Silver → Gold)
5. Training pipeline (Gold → Models)

## Usage Examples

### Schedule Feature Pipeline Daily
```bash
# Cron job to create fresh features daily
0 2 * * * cd /path/to/project && uv run python ml_classification/pipelines/feature_pipeline.py
```

### Retrain Models Weekly
```bash
# Use pre-computed features, train new models
0 4 * * 0 cd /path/to/project && uv run python ml_classification/pipelines/training_pipeline.py
```

### Batch Inference Hourly
```bash
# Score new data every hour
0 * * * * cd /path/to/project && uv run python ml_classification/pipelines/inference_pipeline.py \
  models:/default-payment-randomforestclassifier/latest \
  s3://bucket/new_data.parquet \
  s3://bucket/predictions/$(date +\%Y\%m\%d_\%H).parquet
```

## Key Design Principles

### 1. Separation of Concerns
- Feature engineering is isolated from training
- Training is isolated from inference
- Each pipeline has a single responsibility

### 2. Feature Consistency
- Same feature code used in training and inference
- Prevents train-serve skew
- Features are reusable across models

### 3. Version Tracking
- Features are versioned with metadata
- Models track which feature version they used
- Full lineage from data to predictions

### 4. Independent Scheduling
- Pipelines run on different schedules
- Feature pipeline: Daily (when new data arrives)
- Training pipeline: Weekly/monthly (when needed)
- Inference pipeline: Hourly/on-demand (for scoring)

### 5. Testability
- Each pipeline can be tested independently
- Unit tests for feature engineering
- Integration tests for pipeline compatibility
- No cross-pipeline dependencies in tests

## Feature Engineering Module

The `ml_classification.features.engineering` module contains all feature transformations:

```python
from ml_classification.features.engineering import engineer_features

# Apply all feature engineering
df_with_features = engineer_features(raw_df)
```

**Available Functions**:
- `create_age_bins()`: Age categorization
- `create_bill_trend()`: Bill amount trends
- `create_pay_ratio()`: Payment to bill ratio
- `create_utilization()`: Credit utilization
- `engineer_features()`: Apply all transformations
- `get_feature_names()`: Get list of engineered features

## Preprocessing Module

The `ml_classification.features.preprocessing` module handles scaling and encoding:

```python
from ml_classification.features.preprocessing import (
    build_preprocessor,
    get_preprocessing_config,
    save_preprocessing_config,
    load_preprocessing_config
)

# Build preprocessor from data
preprocessor = build_preprocessor(X_train)

# Save configuration
config = get_preprocessing_config(X_train)
save_preprocessing_config(config, "s3://bucket/preprocessing_config.json")

# Load and rebuild preprocessor
config = load_preprocessing_config("s3://bucket/preprocessing_config.json")
preprocessor = build_preprocessor_from_config(config)
```

## Migration from Legacy Code

### Old Commands → New Commands

| Legacy Command | New Command | Notes |
|---------------|-------------|-------|
| `make.py gold` | `make.py feature-pipeline` | Use new pipeline |
| `make.py train` | `make.py training-pipeline` | Use new pipeline |
| `make.py pipeline` | `make.py full-pipeline` | Use new pipeline |
| Batch inference (old) | `make.py inference-pipeline` | Now includes feature engineering |

### Code Migration

**Before** (Training):
```python
# Old: Feature engineering in training code
from ml_classification.modeling.data import load_data, build_preprocessor
X_train, X_test, y_train, y_test, metadata = load_data(path, target)
preprocessor = build_preprocessor(X_train)
```

**After** (Training):
```python
# New: Load pre-computed features
from ml_classification.pipelines.training_pipeline import load_features_with_metadata
X_train, X_test, y_train, y_test, metadata = load_features_with_metadata(path, target)
```

**Before** (Inference):
```python
# Old: Manual feature handling
X = pd.read_parquet(path)
X = X.drop(columns=['target'])
predictions = model.predict(X)
```

**After** (Inference):
```python
# New: Feature engineering included
from ml_classification.pipelines.inference_pipeline import run_batch_inference
run_batch_inference(model_uri, input_path, output_path)
```

## Testing

Run tests for the new modules:

```bash
# Test feature engineering
uv run pytest tests/features/test_engineering.py

# Test preprocessing
uv run pytest tests/features/test_preprocessing.py

# Test pipeline integration
uv run pytest tests/pipelines/test_integration.py

# Run all tests
uv run pytest tests/
```

## Benefits Summary

✅ **Independent Scheduling**: Run pipelines at different frequencies

✅ **No Train-Serve Skew**: Same feature code everywhere

✅ **Faster Iteration**: Retrain without recomputing features

✅ **Better Versioning**: Track feature versions with models

✅ **Clearer Code**: Single responsibility per pipeline

✅ **Easier Testing**: Test components independently

✅ **Better Collaboration**: Teams can own different pipelines

✅ **Production Ready**: Supports real-world deployment patterns
