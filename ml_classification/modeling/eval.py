from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def precision_recall_at_k(y_true, y_scores, k=0.1):
    """Calculate precision and recall at top k% of predictions."""
    threshold = np.quantile(y_scores, 1 - k)
    y_pred = (y_scores >= threshold).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return precision, recall

def plot_confusion_matrix(y_true, y_pred):
    """Generate and plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return cm

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation with multiple metrics."""
    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba),
    }

    # Calculate precision and recall at different thresholds
    for k in [0.01, 0.05, 0.1, 0.2]:
        precision_k, recall_k = precision_recall_at_k(y_test, y_proba, k=k)
        metrics[f"precision_at_{int(k*100)}"] = precision_k
        metrics[f"recall_at_{int(k*100)}"] = recall_k

    # Generate confusion matrix
    cm = plot_confusion_matrix(y_test, y_pred)
    
    # Log all metrics
    logger.info("\nModel Performance Metrics:")
    logger.info("-" * 50)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.3f}")
    
    # Log classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))

    return metrics, cm, y_proba