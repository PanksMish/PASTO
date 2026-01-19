"""
Visualization utilities for PASTO
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_training_curves(
    history: List[Dict],
    metrics: List[str] = ['loss', 'auc_pr', 'retention_rate'],
    save_path: Optional[str] = None
):
    """
    Plot training curves
    
    Args:
        history: List of epoch dictionaries
        metrics: Metrics to plot
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        train_values = []
        val_values = []
        
        for epoch_dict in history:
            if metric in epoch_dict.get('train_metrics', {}):
                train_values.append(epoch_dict['train_metrics'][metric])
            if metric in epoch_dict.get('val_metrics', {}):
                val_values.append(epoch_dict['val_metrics'][metric])
        
        epochs = range(1, len(train_values) + 1)
        
        if train_values:
            ax.plot(epochs, train_values, 'o-', label='Train', linewidth=2)
        if val_values:
            ax.plot(epochs, val_values, 's-', label='Val', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} vs Epoch', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ['Retained', 'Dropout'],
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot precision-recall curve
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        save_path: Path to save figure
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved PR curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        save_path: Path to save figure
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 20,
    save_path: Optional[str] = None
):
    """
    Plot feature importance
    
    Args:
        importance: Feature importance scores
        feature_names: Names of features
        top_k: Number of top features to show
        save_path: Path to save figure
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importance))]
    
    # Get top-k features
    top_indices = np.argsort(importance)[-top_k:][::-1]
    top_importance = importance[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_importance)), top_importance, color='steelblue')
    plt.yticks(range(len(top_importance)), top_names)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_k} Feature Importance', fontsize=14)
    plt.gca().invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_retention_over_time(
    retention_rates: List[float],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot retention rate over time
    
    Args:
        retention_rates: List of retention rates over time
        labels: Model labels
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(retention_rates[0], list):
        # Multiple models
        for i, rates in enumerate(retention_rates):
            label = labels[i] if labels else f'Model {i+1}'
            plt.plot(rates, 'o-', linewidth=2, label=label)
    else:
        # Single model
        plt.plot(retention_rates, 'o-', linewidth=2, label='Retention Rate')
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Retention Rate', fontsize=12)
    plt.title('Retention Rate Over Time', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved retention plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_cumulative_regret(
    regrets: List[np.ndarray],
    labels: List[str],
    save_path: Optional[str] = None
):
    """
    Plot cumulative regret comparison
    
    Args:
        regrets: List of cumulative regret arrays
        labels: Model labels
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    for regret, label in zip(regrets, labels):
        plt.plot(regret, linewidth=2, label=label)
    
    plt.xlabel('Decision Round', fontsize=12)
    plt.ylabel('Cumulative Regret', fontsize=12)
    plt.title('Cumulative Regret Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved regret plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_names: List[str] = ['auc_pr', 'recall', 'f2', 'retention_rate'],
    save_path: Optional[str] = None
):
    """
    Plot model comparison across metrics
    
    Args:
        metrics: Dictionary of {model_name: {metric: value}}
        metric_names: Metrics to compare
        save_path: Path to save figure
    """
    models = list(metrics.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        values = [metrics[model].get(m, 0) for m in metric_names]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization utilities...")
    
    # Create test directory
    save_dir = Path("test_plots")
    save_dir.mkdir(exist_ok=True)
    
    # Test data
    y_true = np.random.randint(0, 2, 1000)
    y_scores = np.random.rand(1000)
    y_pred = (y_scores > 0.5).astype(int)
    
    # Test plots
    plot_precision_recall_curve(y_true, y_scores, save_dir / "pr_curve.png")
    plot_roc_curve(y_true, y_scores, save_dir / "roc_curve.png")
    plot_confusion_matrix(y_true, y_pred, save_path=save_dir / "confusion_matrix.png")
    
    # Test retention plot
    retention = [0.8, 0.78, 0.76, 0.75, 0.73, 0.72, 0.71]
    plot_retention_over_time(retention, save_path=save_dir / "retention.png")
    
    print(f"\n✓ All test plots saved to {save_dir}")
