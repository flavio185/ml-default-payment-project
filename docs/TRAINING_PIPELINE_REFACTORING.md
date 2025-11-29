# Training Pipeline Refactoring - Externalized Responsibilities

## Problem: Training Pipeline Was Doing Too Much

The original `training_pipeline.py` had **7 different responsibilities**:

```python
# ❌ TOO MANY RESPONSIBILITIES IN ONE FILE
training_pipeline.py:
  - Data loading from S3
  - Metadata extraction
  - Train/test splitting
  - Preprocessor building
  - Pipeline creation
  - Model training
  - Model evaluation
  - MLflow logging
  - Artifact management
```

This violates the **Single Responsibility Principle** and makes:
- ❌ Testing difficult (too many things to mock)
- ❌ Reusability hard (can't use parts independently)
- ❌ Maintenance complex (changes affect multiple concerns)
- ❌ Extension difficult (adding new features touches everything)

## Solution: Externalized Responsibilities

We've created **specialized modules** with clear responsibilities:

### New Architecture

```
ml_classification/modeling/
├── data_loader.py         # Responsibility: Load and split data
├── pipeline_builder.py    # Responsibility: Create sklearn pipelines
├── trainer.py             # Responsibility: Train and evaluate models
├── mlflow_logger.py       # Responsibility: Log to MLflow
├── models.py              # Responsibility: Model definitions (existing)
└── eval.py                # Responsibility: Metrics calculation (existing)
```

### 1. Data Loader (`data_loader.py`)

**Single Responsibility**: Load features and metadata from Gold layer

```python
from ml_classification.modeling.data_loader import load_features

X_train, X_test, y_train, y_test, metadata = load_features(
    features_path="s3://bucket/gold/features.parquet",
    target_col="default_payment_next_month"
)
```

**What it does:**
- ✅ Loads features from S3
- ✅ Loads feature metadata
- ✅ Splits train/test with stratification
- ✅ Returns clean data ready for training

**What it doesn't do:**
- ❌ No training
- ❌ No preprocessing (that's in pipeline_builder)
- ❌ No MLflow logging

### 2. Pipeline Builder (`pipeline_builder.py`)

**Single Responsibility**: Create sklearn pipelines with preprocessing

```python
from ml_classification.modeling.pipeline_builder import create_sklearn_pipeline

pipeline = create_sklearn_pipeline(X_train, model)
```

**What it does:**
- ✅ Builds preprocessor from training data
- ✅ Combines preprocessor + model into Pipeline
- ✅ Returns ready-to-train pipeline

**What it doesn't do:**
- ❌ No data loading
- ❌ No training (that's in trainer)
- ❌ No evaluation

### 3. Trainer (`trainer.py`)

**Single Responsibility**: Train and evaluate models

```python
from ml_classification.modeling.trainer import train_and_evaluate

pipeline, metrics, cm, y_proba = train_and_evaluate(
    pipeline, X_train, y_train, X_test, y_test
)
```

**What it does:**
- ✅ Trains sklearn pipeline
- ✅ Evaluates on test set
- ✅ Returns metrics and predictions

**What it doesn't do:**
- ❌ No MLflow logging (that's in mlflow_logger)
- ❌ No data loading
- ❌ No pipeline creation

### 4. MLflow Logger (`mlflow_logger.py`)

**Single Responsibility**: Log training runs to MLflow

```python
from ml_classification.modeling.mlflow_logger import MLflowExperimentLogger

mlflow_logger = MLflowExperimentLogger("my-experiment")
mlflow_logger.log_training_run(
    pipeline=trained_pipeline,
    X_train=X_train,
    X_test=X_test,
    metrics=metrics,
    confusion_matrix=cm,
    feature_metadata=metadata
)
```

**What it does:**
- ✅ Logs parameters to MLflow
- ✅ Logs metrics to MLflow
- ✅ Logs artifacts (confusion matrix, metadata)
- ✅ Registers models
- ✅ Tracks feature versions

**What it doesn't do:**
- ❌ No training
- ❌ No evaluation
- ❌ No data loading

## The Refactored Training Pipeline V2

Now the training pipeline is **simple and clean** - it just orchestrates:

```python
# training_pipeline_v2.py

def run_training_pipeline(...):
    # 1. Load data
    X_train, X_test, y_train, y_test, metadata = load_features(...)

    # 2. Initialize MLflow logger
    mlflow_logger = MLflowExperimentLogger(experiment_name)

    # 3. For each model:
    for model in models:
        # 3a. Create pipeline
        pipeline = create_sklearn_pipeline(X_train, model)

        # 3b. Train and evaluate
        pipeline, metrics, cm, y_proba = train_and_evaluate(
            pipeline, X_train, y_train, X_test, y_test
        )

        # 3c. Log to MLflow
        mlflow_logger.log_training_run(...)
```

**That's it!** ~50 lines instead of ~200 lines.

## Comparison: Before vs After

### Before (training_pipeline.py)

```python
# ❌ 200+ lines doing everything

def load_feature_metadata(features_path):
    # Metadata loading logic
    ...

def load_features_with_metadata(...):
    # Data loading logic
    # Splitting logic
    # Metadata enrichment
    ...

def create_pipeline(X_train, model):
    # Preprocessor building
    # Pipeline creation
    ...

def log_model_run(...):
    # Parameter logging
    # Metric logging
    # Artifact logging
    # Model registration
    ...

def run_training_pipeline(...):
    # Orchestration mixed with implementation
    ...
```

### After (training_pipeline_v2.py)

```python
# ✅ ~50 lines, just orchestration

from ml_classification.modeling.data_loader import load_features
from ml_classification.modeling.pipeline_builder import create_sklearn_pipeline
from ml_classification.modeling.trainer import train_and_evaluate
from ml_classification.modeling.mlflow_logger import MLflowExperimentLogger

def run_training_pipeline(...):
    # Clean orchestration
    data = load_features(...)
    logger = MLflowExperimentLogger(...)

    for model in models:
        pipeline = create_sklearn_pipeline(...)
        results = train_and_evaluate(...)
        logger.log_training_run(...)
```

## Benefits

### 1. **Better Testability**

```python
# Test each component independently

def test_data_loader():
    # Mock S3, test loading logic
    X_train, X_test, y_train, y_test, metadata = load_features(...)
    assert X_train.shape[0] > 0

def test_pipeline_builder():
    # Test pipeline creation without data loading
    pipeline = create_sklearn_pipeline(X_train, model)
    assert isinstance(pipeline, Pipeline)

def test_trainer():
    # Test training without MLflow
    pipeline, metrics, cm, y_proba = train_and_evaluate(...)
    assert 'accuracy' in metrics

def test_mlflow_logger():
    # Test logging without training
    with mock_mlflow():
        logger.log_training_run(...)
```

### 2. **Better Reusability**

```python
# Reuse components in different contexts

# Use data_loader in notebooks
X_train, X_test, y_train, y_test, _ = load_features(path, target)

# Use pipeline_builder for experimentation
pipeline = create_sklearn_pipeline(X_train, my_custom_model)

# Use trainer for quick experiments
results = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test)

# Use mlflow_logger for custom workflows
logger = MLflowExperimentLogger("my-experiment")
logger.log_training_run(...)
```

### 3. **Easier Extension**

```python
# Add a new logging backend? Just create new logger
class WandBLogger:
    def log_training_run(...):
        # Log to Weights & Biases instead
        ...

# Add a new data source? Just create new loader
def load_features_from_delta(table_name, target_col):
    # Load from Delta Lake
    ...

# Add cross-validation? Extend trainer
def train_with_cross_validation(...):
    # Add CV logic
    ...
```

### 4. **Clearer Code Organization**

Each file has a **clear purpose**:

```
Need to change how data is loaded?
→ Edit data_loader.py

Need to change preprocessing?
→ Edit pipeline_builder.py

Need to change training logic?
→ Edit trainer.py

Need to change MLflow logging?
→ Edit mlflow_logger.py
```

## How to Use

### Option 1: Use the new pipeline directly

```bash
# Use the refactored version
python ml_classification/pipelines/training_pipeline_v2.py
```

### Option 2: Use individual components

```python
# In a notebook or script
from ml_classification.modeling.data_loader import load_features
from ml_classification.modeling.pipeline_builder import create_sklearn_pipeline
from ml_classification.modeling.trainer import train_and_evaluate

# Load data
X_train, X_test, y_train, y_test, metadata = load_features(
    "s3://bucket/gold/features.parquet",
    "default_payment_next_month"
)

# Create custom pipeline
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=200)
pipeline = create_sklearn_pipeline(X_train, model)

# Train
pipeline, metrics, cm, y_proba = train_and_evaluate(
    pipeline, X_train, y_train, X_test, y_test
)

print(f"Accuracy: {metrics['accuracy']}")
```

### Option 3: Mix and match

```python
# Use some new components, customize others
from ml_classification.modeling.data_loader import load_features
from ml_classification.modeling.mlflow_logger import MLflowExperimentLogger

# Load data (using new component)
data = load_features(...)

# Custom training loop
for model in my_custom_models:
    # Your custom logic
    ...

    # Use MLflow logger for consistency
    mlflow_logger = MLflowExperimentLogger("experiment")
    mlflow_logger.log_training_run(...)
```

## Migration Path

### Phase 1: Use Both (Current State)
- Keep `training_pipeline.py` for backward compatibility
- Use `training_pipeline_v2.py` for new development

### Phase 2: Gradual Migration
- Update `make.py` to use v2
- Update CI/CD to use v2
- Deprecate v1

### Phase 3: Remove Old Version
- Delete `training_pipeline.py`
- Rename `training_pipeline_v2.py` → `training_pipeline.py`

## Update make.py (Optional)

```python
# Add to make.py
@app.command(name="training-pipeline-v2")
def training_pipeline_v2():
    """Run refactored training pipeline with externalized responsibilities."""
    console.print("[bold magenta]>>> TRAINING PIPELINE V2 STARTED[/bold magenta]")
    requirements()
    _run_command(["uv", "run", "python", "ml_classification/pipelines/training_pipeline_v2.py"])
    console.print("[bold green]>>> TRAINING PIPELINE V2 COMPLETED[/bold green]")
```

## Summary

### Original Training Pipeline
- ❌ 7 responsibilities in one file
- ❌ Hard to test
- ❌ Hard to reuse
- ❌ Hard to extend

### Refactored Training Pipeline
- ✅ Single responsibility per module
- ✅ Easy to test (mock one thing at a time)
- ✅ Easy to reuse (import what you need)
- ✅ Easy to extend (add new implementations)
- ✅ Clear separation of concerns
- ✅ Better code organization

### The New Modules

| Module | Responsibility | LOC |
|--------|---------------|-----|
| `data_loader.py` | Load and split data | ~70 |
| `pipeline_builder.py` | Create sklearn pipelines | ~50 |
| `trainer.py` | Train and evaluate | ~60 |
| `mlflow_logger.py` | Log to MLflow | ~120 |
| `training_pipeline_v2.py` | Orchestrate everything | ~80 |

**Total**: ~380 lines across 5 well-organized files
vs
**Before**: ~200 lines in 1 monolithic file

More lines, but **much better organized and maintainable**! 🎉
