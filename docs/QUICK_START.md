# Quick Start Guide - New Pipeline Architecture

## What's New?

Your ML project now has **three independent pipelines**:

1. 🔧 **Feature Pipeline** - Creates reusable features
2. 🤖 **Training Pipeline** - Trains models with feature versioning
3. 🎯 **Inference Pipeline** - Predicts with consistent feature engineering

## 5-Minute Quick Start

### 1. Run the Feature Pipeline

Create versioned features from your Silver layer data:

```bash
python make.py feature-pipeline
```

**What it does:**
- Loads data from `s3://bucket/silver/credit_card_default.parquet`
- Applies feature engineering (age bins, bill trends, pay ratios, etc.)
- Saves to `s3://bucket/gold/credit_card_default_features.parquet`
- Creates metadata with feature versions and preprocessing config

**Output Files:**
- `credit_card_default_features.parquet` - The features
- `credit_card_default_features_metadata.json` - Feature metadata
- `credit_card_default_features_preprocessing_config.json` - Preprocessing config

### 2. Run the Training Pipeline

Train models using the pre-computed features:

```bash
python make.py training-pipeline
```

**What it does:**
- Loads features from Gold layer with metadata
- Trains multiple models (LogisticRegression, RandomForest)
- Evaluates and logs everything to MLflow
- Registers models with feature version tracking

**Output:**
- MLflow models: `models:/default-payment-logisticregression/1`
- MLflow models: `models:/default-payment-randomforestclassifier/1`
- Metrics, plots, and artifacts in MLflow

### 3. Run Batch Inference

Score new data with automatic feature engineering:

```bash
python make.py inference-pipeline \
  models:/default-payment-randomforestclassifier/1 \
  s3://bucket/new_data.parquet \
  s3://bucket/predictions.parquet
```

**What it does:**
- Loads raw data
- Applies **same feature engineering** as training (no skew!)
- Loads model from MLflow
- Generates predictions with probabilities
- Saves results

**Output:**
- Predictions parquet with `prediction`, `probability`, `inference_timestamp` columns

## Run Everything Together

```bash
# Full pipeline: Bronze → Silver → Validate → Features → Training
python make.py full-pipeline
```

## Code Examples

### Using Feature Engineering in Your Code

```python
from ml_classification.features.engineering import engineer_features

# Apply all feature transformations
df_with_features = engineer_features(raw_df)
```

### Using Preprocessing

```python
from ml_classification.features.preprocessing import (
    build_preprocessor,
    get_preprocessing_config
)

# Build preprocessor
preprocessor = build_preprocessor(X_train)

# Save config for later
config = get_preprocessing_config(X_train)
```

### Running Inference Programmatically

```python
from ml_classification.pipelines.inference_pipeline import run_batch_inference

run_batch_inference(
    model_uri="models:/default-payment-randomforestclassifier/1",
    input_path="s3://bucket/data.parquet",
    output_path="s3://bucket/predictions.parquet"
)
```

## Directory Structure

```
ml_classification/
├── features/              # NEW: Feature engineering modules
│   ├── engineering.py     # All feature transformations
│   └── preprocessing.py   # Scaling, encoding, config
│
├── pipelines/             # NEW: Pipeline orchestration
│   ├── feature_pipeline.py
│   ├── training_pipeline.py
│   └── inference_pipeline.py
│
├── modeling/              # Existing model code
└── serving/               # Existing serving code
```

## Testing

Run the new tests:

```bash
# Test feature engineering
pytest tests/features/test_engineering.py

# Test preprocessing
pytest tests/features/test_preprocessing.py

# Test pipeline integration
pytest tests/pipelines/test_integration.py

# Run all tests
pytest tests/
```

## Common Tasks

### Add a New Feature

Edit `ml_classification/features/engineering.py`:

```python
def create_my_new_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create my new feature."""
    df['my_feature'] = df['col1'] / df['col2']
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all features."""
    df = create_age_bins(df)
    df = create_bill_trend(df)
    df = create_pay_ratio(df)
    df = create_utilization(df)
    df = create_my_new_feature(df)  # Add here
    return df

def get_feature_names() -> list[str]:
    """Get all feature names."""
    return ["age_bin", "bill_trend", "pay_ratio", "utilization", "my_feature"]
```

Then:
1. Run feature pipeline: `python make.py feature-pipeline`
2. Run training pipeline: `python make.py training-pipeline`
3. New models will use your new feature!

### Retrain with Existing Features

```bash
# Just retrain, don't recreate features
python make.py training-pipeline
```

### Check Feature Version Used by Model

```python
import mlflow

client = mlflow.MlflowClient()
model_version = client.get_model_version("default-payment-logisticregression", "1")
run = client.get_run(model_version.run_id)

# Get feature version
feature_version = run.data.params.get('feature_version')
print(f"Model trained with feature version: {feature_version}")
```

## Migration from Old Commands

| Old Command | New Command | Why Change? |
|------------|-------------|-------------|
| `make.py gold` | `make.py feature-pipeline` | Better separation, versioning |
| `make.py train` | `make.py training-pipeline` | Uses pre-computed features |
| `make.py pipeline` | `make.py full-pipeline` | More descriptive name |

**Note:** Old commands still work with deprecation warnings.

## Typical Workflow

### Development
```bash
# 1. Create features once
python make.py feature-pipeline

# 2. Iterate on models (fast!)
python make.py training-pipeline  # First model
# ... edit model config ...
python make.py training-pipeline  # Second model
# ... repeat as needed ...
```

### Production
```bash
# Daily: Update features when new data arrives
0 4 * * * python make.py feature-pipeline

# Weekly: Retrain models
0 5 * * 0 python make.py training-pipeline

# Hourly: Batch inference
0 * * * * python make.py inference-pipeline \
  models:/default-payment/latest \
  s3://bucket/input.parquet \
  s3://bucket/output/$(date +\%Y\%m\%d_\%H).parquet
```

## Benefits You Get

✅ **Independent Scheduling**
- Run feature pipeline daily
- Run training weekly
- Run inference hourly

✅ **No Train-Serve Skew**
- Same feature code everywhere
- Consistent transformations

✅ **Faster Development**
- Create features once
- Iterate on models quickly
- No wasted computation

✅ **Better Tracking**
- Feature versions tracked
- Full model lineage
- Reproducible results

✅ **Production Ready**
- Follows ML best practices
- Scalable architecture
- Easy to maintain

## Documentation

- **Full Architecture**: [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md)
- **Data Flow**: [PIPELINE_FLOW.md](PIPELINE_FLOW.md)
- **Migration Guide**: [../REFACTORING_SUMMARY.md](../REFACTORING_SUMMARY.md)

## Troubleshooting

### "Module not found" errors

Make sure you're running from the project root:
```bash
cd /path/to/ml-default-payment-project
python make.py feature-pipeline
```

### Feature pipeline fails

Check that Silver layer data exists:
```bash
python make.py silver
python make.py validate
python make.py feature-pipeline
```

### Training pipeline fails

Check that Gold layer features exist:
```bash
python make.py feature-pipeline
python make.py training-pipeline
```

### Inference fails

Make sure you have a registered model:
```bash
python make.py training-pipeline
# Then use the model URI shown in the output
```

## Need Help?

- Check logs for detailed error messages
- Run tests: `pytest tests/`
- Read full docs in `docs/PIPELINE_ARCHITECTURE.md`
- Review examples in `tests/` directory

## What's Next?

1. ✅ Run the pipelines and verify they work
2. 📊 Check MLflow UI to see your models
3. 🔧 Add custom features to `engineering.py`
4. 📈 Monitor pipeline runs in production
5. 🚀 Integrate with your orchestration tool (Airflow, Prefect, etc.)

Happy ML Engineering! 🎉
