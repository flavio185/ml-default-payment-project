# Complete ML Architecture - Final Structure

## Overview

The ML project now has a **clean, modular architecture** following software engineering best practices with clear separation of concerns.

## Complete Directory Structure

```
ml_classification/
│
├── features/                      # Feature Engineering Layer
│   ├── engineering.py            # Feature transformations
│   └── preprocessing.py          # Preprocessing & config management
│
├── pipelines/                     # Orchestration Layer
│   ├── feature_pipeline.py       # Silver → Gold orchestration
│   ├── training_pipeline.py      # Training orchestration (v1)
│   ├── training_pipeline_v2.py   # Training orchestration (v2 - refactored)
│   └── inference_pipeline.py     # Inference orchestration
│
├── modeling/                      # ML Core Layer
│   ├── data_loader.py            # NEW: Data loading responsibility
│   ├── pipeline_builder.py       # NEW: Pipeline creation responsibility
│   ├── trainer.py                # NEW: Training responsibility
│   ├── mlflow_logger.py          # NEW: MLflow logging responsibility
│   ├── models.py                 # Model definitions
│   ├── eval.py                   # Evaluation metrics
│   ├── train.py                  # Legacy training (kept for compatibility)
│   ├── data.py                   # Legacy data utilities
│   └── predict.py                # Ad-hoc predictions
│
├── serving/                       # Serving Layer
│   ├── app.py                    # FastAPI online inference
│   └── batch_inference.py        # Batch inference wrapper
│
└── config.py                      # Global configuration
```

## Architectural Layers

### Layer 1: Feature Engineering
**Responsibility**: Transform raw data into ML features

```
features/
├── engineering.py      → Domain-specific feature creation
└── preprocessing.py    → Scaling, encoding, config management
```

**Key Principle**: Single source of truth for features
- Same code used in training and inference
- No train-serve skew
- Reusable across all pipelines

### Layer 2: ML Core
**Responsibility**: Core ML operations (modular, testable)

```
modeling/
├── data_loader.py      → Load and split data
├── pipeline_builder.py → Create sklearn pipelines
├── trainer.py          → Train and evaluate models
├── mlflow_logger.py    → Log to MLflow
├── models.py           → Model definitions
└── eval.py             → Metrics calculation
```

**Key Principle**: Single Responsibility
- Each module does ONE thing well
- Easy to test individually
- Easy to reuse in different contexts
- Easy to extend with new implementations

### Layer 3: Pipeline Orchestration
**Responsibility**: Coordinate ML workflows

```
pipelines/
├── feature_pipeline.py     → Orchestrate feature engineering
├── training_pipeline_v2.py → Orchestrate model training
└── inference_pipeline.py   → Orchestrate batch inference
```

**Key Principle**: Thin orchestration layer
- Just coordinates the components
- No business logic
- Clear, readable workflow
- Easy to understand data flow

### Layer 4: Serving
**Responsibility**: Expose models for predictions

```
serving/
├── app.py              → Online inference (FastAPI)
└── batch_inference.py  → Batch inference wrapper
```

**Key Principle**: Reuse inference pipeline
- Consistent feature engineering
- No duplicate code
- Easy to deploy

## Responsibility Matrix

| Component | Data Loading | Feature Eng | Preprocessing | Training | Evaluation | MLflow | Pipeline Creation |
|-----------|-------------|-------------|---------------|----------|------------|--------|------------------|
| `features/engineering.py` | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `features/preprocessing.py` | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `modeling/data_loader.py` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `modeling/pipeline_builder.py` | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| `modeling/trainer.py` | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| `modeling/mlflow_logger.py` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `modeling/models.py` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `modeling/eval.py` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `pipelines/feature_pipeline.py` | 🔄 | 🔄 | 🔄 | ❌ | ❌ | ❌ | ❌ |
| `pipelines/training_pipeline_v2.py` | 🔄 | ❌ | 🔄 | 🔄 | 🔄 | 🔄 | 🔄 |
| `pipelines/inference_pipeline.py` | 🔄 | 🔄 | ❌ | ❌ | ❌ | ❌ | ❌ |

Legend:
- ✅ = Primary responsibility (implements logic)
- 🔄 = Orchestrates (delegates to other modules)
- ❌ = Not responsible

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING                           │
│  Bronze → Silver → Validate                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE PIPELINE                                │
│                                                              │
│  Load Silver → engineer_features() → Save Gold + Metadata   │
│                                                              │
│  Components Used:                                            │
│  • features/engineering.py                                   │
│  • features/preprocessing.py                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            TRAINING PIPELINE V2                              │
│                                                              │
│  1. data_loader.load_features()                              │
│  2. pipeline_builder.create_sklearn_pipeline()               │
│  3. trainer.train_and_evaluate()                             │
│  4. mlflow_logger.log_training_run()                         │
│                                                              │
│  Components Used:                                            │
│  • modeling/data_loader.py                                   │
│  • modeling/pipeline_builder.py                              │
│  • modeling/trainer.py                                       │
│  • modeling/mlflow_logger.py                                 │
│  • modeling/models.py                                        │
│  • modeling/eval.py                                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   MLFLOW REGISTRY                            │
│  Registered Models with Feature Versions                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│             INFERENCE PIPELINE                               │
│                                                              │
│  Load Raw Data → engineer_features() → Load Model → Predict │
│                                                              │
│  Components Used:                                            │
│  • features/engineering.py (same as training!)               │
│  • MLflow model loading                                     │
└─────────────────────────────────────────────────────────────┘
```

## Usage Examples

### 1. Using Feature Engineering

```python
from ml_classification.features.engineering import engineer_features

# Apply all feature transformations
df_with_features = engineer_features(raw_df)
```

### 2. Using Data Loader

```python
from ml_classification.modeling.data_loader import load_features

# Load and split data
X_train, X_test, y_train, y_test, metadata = load_features(
    features_path="s3://bucket/gold/features.parquet",
    target_col="default_payment_next_month"
)
```

### 3. Using Pipeline Builder

```python
from ml_classification.modeling.pipeline_builder import create_sklearn_pipeline
from sklearn.ensemble import RandomForestClassifier

# Create pipeline
model = RandomForestClassifier(n_estimators=100)
pipeline = create_sklearn_pipeline(X_train, model)
```

### 4. Using Trainer

```python
from ml_classification.modeling.trainer import train_and_evaluate

# Train and evaluate
pipeline, metrics, cm, y_proba = train_and_evaluate(
    pipeline, X_train, y_train, X_test, y_test
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
```

### 5. Using MLflow Logger

```python
from ml_classification.modeling.mlflow_logger import MLflowExperimentLogger

# Log to MLflow
logger = MLflowExperimentLogger("my-experiment")
logger.log_training_run(
    pipeline=pipeline,
    X_train=X_train,
    X_test=X_test,
    metrics=metrics,
    confusion_matrix=cm,
    feature_metadata=metadata
)
```

### 6. Running Complete Pipelines

```bash
# Feature pipeline
python make.py feature-pipeline

# Training pipeline (v2 - refactored)
python ml_classification/pipelines/training_pipeline_v2.py

# Inference pipeline
python make.py inference-pipeline \
  models:/default-payment-randomforestclassifier/1 \
  s3://bucket/input.parquet \
  s3://bucket/output.parquet
```

## Testing Strategy

### Unit Tests (Test individual modules)

```
tests/
├── features/
│   ├── test_engineering.py       # Test feature functions
│   └── test_preprocessing.py     # Test preprocessing
│
├── ml_classification/modeling/
│   ├── test_pipeline_builder.py  # Test pipeline creation
│   ├── test_trainer.py           # Test training logic
│   └── test_mlflow_logger.py     # Test logging (to add)
│
└── pipelines/
    └── test_integration.py       # Test pipeline integration
```

### Integration Tests (Test pipelines end-to-end)

```bash
# Test feature pipeline
pytest tests/pipelines/test_integration.py::test_feature_consistency_train_inference

# Test training pipeline
pytest tests/ml_classification/modeling/test_trainer.py::test_train_and_evaluate
```

## Benefits Achieved

### ✅ Separation of Concerns
- Each module has ONE responsibility
- Easy to understand what each file does
- Changes are localized

### ✅ Testability
- Test each component independently
- Mock only what you need
- Fast, focused tests

### ✅ Reusability
- Import only what you need
- Use components in different contexts
- No tight coupling

### ✅ Extensibility
- Add new features → Edit `engineering.py`
- Add new models → Edit `models.py`
- Add new logging backend → Create new logger
- Add new data source → Create new loader

### ✅ Maintainability
- Clear file structure
- Easy to find code
- Easy to onboard new developers

### ✅ No Train-Serve Skew
- Same feature code everywhere
- Consistent transformations
- Production-ready architecture

## Command Reference

### Individual Pipelines

```bash
# Feature engineering
python make.py feature-pipeline

# Training (original)
python make.py training-pipeline

# Training (refactored - recommended)
python ml_classification/pipelines/training_pipeline_v2.py

# Inference
python make.py inference-pipeline <model_uri> <input> <output>
```

### Full Pipeline

```bash
# Run everything
python make.py full-pipeline
```

### Testing

```bash
# Test features
pytest tests/features/

# Test modeling components
pytest tests/ml_classification/modeling/

# Test pipelines
pytest tests/pipelines/

# Test everything
pytest tests/
```

## Migration Path

### Phase 1: Current State ✅
- Old pipelines work (backward compatible)
- New modules available for use
- Both v1 and v2 training pipelines exist

### Phase 2: Gradual Adoption
- Start using v2 training pipeline in development
- Update notebooks to use new modules
- Add more tests

### Phase 3: Full Migration
- Update CI/CD to use v2
- Update documentation
- Deprecate v1 (optional)

## Next Steps

1. **Try the new modules**: Use individual components in notebooks
2. **Run training pipeline v2**: Test the refactored pipeline
3. **Add your own models**: Extend `models.py` and use with new architecture
4. **Add tests**: Write tests for your custom logic
5. **Extend as needed**: Add new loaders, loggers, or trainers

## Summary

This architecture provides:

🎯 **Clear Separation**: Each module has a single responsibility

🧪 **Testable**: Easy to test each component independently

🔄 **Reusable**: Use components in different contexts

🚀 **Extensible**: Easy to add new features

📦 **Maintainable**: Easy to understand and modify

✨ **Production-Ready**: Follows ML engineering best practices

**You now have a professional, scalable ML architecture!** 🎉
