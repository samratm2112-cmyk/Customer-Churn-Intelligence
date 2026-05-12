import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

sns.set_theme(style='darkgrid')


def compute_metrics(model: Any, X_test, y_test) -> dict[str, Any]:
    """Compute evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("Computed metrics - accuracy: %.4f, precision: %.4f, recall: %.4f, f1: %.4f", accuracy, precision, recall, f1)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report,
        'confusion_matrix': cm,
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str,
    save_path: Path | None = None,
) -> None:
    """Plot a confusion matrix with enhanced aesthetics."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        annot_kws={'size': 14},
        xticklabels=['No churn', 'Churn'],
        yticklabels=['No churn', 'Churn'],
    )
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    title: str,
    save_path: Path | None = None,
    top_n: int = 12,
) -> None:
    """Plot feature importance for tree-based models or logistic regression coefficients."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = abs(model.coef_.ravel())
    else:
        logger.warning("Model type does not expose feature importance.")
        return

    indices = importances.argsort()[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette='rocket')
    plt.title(title, fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results: dict[str, dict[str, float]], save_path: Path | None = None) -> None:
    """Plot model comparison chart for accuracy, precision, recall, and F1."""
    labels = list(results.keys())
    accuracy = [results[name]['accuracy'] for name in labels]
    precision = [results[name]['precision'] for name in labels]
    recall = [results[name]['recall'] for name in labels]
    f1 = [results[name]['f1'] for name in labels]

    x = range(len(labels))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar([p - width * 1.5 for p in x], accuracy, width=width, label='Accuracy', color='#2a9d8f')
    plt.bar([p - width * 0.5 for p in x], precision, width=width, label='Precision', color='#e9c46a')
    plt.bar([p + width * 0.5 for p in x], recall, width=width, label='Recall', color='#f4a261')
    plt.bar([p + width * 1.5 for p in x], f1, width=width, label='F1 Score', color='#264653')

    plt.xticks(x, labels, fontsize=12)
    plt.ylim(0, 1)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=16)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
