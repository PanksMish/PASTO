"""
Evaluation Metrics for PASTO
Implements all metrics from the paper including predictive, policy, cost, and equity metrics
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score
)
from scipy import stats
from typing import Dict, Tuple, Optional, List


def compute_auc_pr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Area Under Precision-Recall Curve
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        
    Returns:
        AUC-PR score
    """
    try:
        return average_precision_score(y_true, y_scores)
    except:
        return 0.0


def compute_auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        
    Returns:
        AUC-ROC score
    """
    try:
        return roc_auc_score(y_true, y_scores)
    except:
        return 0.0


def compute_f_beta_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 2.0
) -> float:
    """
    Compute F-beta score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        beta: Beta parameter (2.0 for F2, favoring recall)
        
    Returns:
        F-beta score
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    if precision + recall == 0:
        return 0.0
    
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    return f_beta


def compute_ks_statistic(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Kolmogorov-Smirnov statistic
    Measures separation between positive and negative classes
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        
    Returns:
        KS statistic
    """
    # Separate scores by class
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0
    
    # Compute KS statistic
    ks_stat, _ = stats.ks_2samp(pos_scores, neg_scores)
    
    return ks_stat


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = 'f2'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        metric: Metric to optimize ('f1', 'f2', 'recall')
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    best_threshold = 0.5
    best_metric_value = 0.0
    
    for threshold, precision, recall in zip(thresholds, precisions[:-1], recalls[:-1]):
        if metric == 'f1':
            if precision + recall > 0:
                current_metric = 2 * (precision * recall) / (precision + recall)
            else:
                current_metric = 0
        elif metric == 'f2':
            if precision + recall > 0:
                current_metric = 5 * (precision * recall) / (4 * precision + recall)
            else:
                current_metric = 0
        elif metric == 'recall':
            current_metric = recall
        else:
            current_metric = precision
        
        if current_metric > best_metric_value:
            best_metric_value = current_metric
            best_threshold = threshold
    
    return best_threshold, best_metric_value


def compute_trajectory_gain(
    trajectory_preds: np.ndarray,
    trajectory_labels: np.ndarray,
    baseline_preds: Optional[np.ndarray] = None,
    discount_factor: float = 0.95
) -> float:
    """
    Compute trajectory gain over baseline
    As defined in Equation 14 of the paper
    
    Args:
        trajectory_preds: Predicted trajectories
        trajectory_labels: Ground truth trajectories
        baseline_preds: Baseline predictions (if None, use labels)
        discount_factor: Discount factor gamma
        
    Returns:
        Trajectory gain
    """
    if baseline_preds is None:
        baseline_preds = trajectory_labels
    
    # Compute improvement
    improvement = trajectory_preds - baseline_preds
    
    # Apply discount if multiple time steps
    if len(improvement.shape) > 1:
        T = improvement.shape[1]
        discounts = np.array([discount_factor**t for t in range(T)])
        improvement = improvement * discounts
    
    # Average gain
    trajectory_gain = improvement.mean()
    
    return trajectory_gain


def compute_intervention_efficiency_ratio(
    retention_gain: float,
    intervention_cost: float
) -> float:
    """
    Compute Intervention Efficiency Ratio (IER)
    As defined in Equation 16 of the paper
    
    Args:
        retention_gain: Improvement in retention rate
        intervention_cost: Total intervention cost
        
    Returns:
        IER value
    """
    if intervention_cost == 0:
        return 0.0
    
    return retention_gain / intervention_cost


def compute_equity_adjusted_retention_gain(
    retention_with_intervention: np.ndarray,
    retention_without_intervention: np.ndarray,
    protected_group_mask: np.ndarray
) -> float:
    """
    Compute Equity-Adjusted Retention Gain (EARG)
    As defined in Equation 15 of the paper
    
    Args:
        retention_with_intervention: Retention indicators with intervention
        retention_without_intervention: Retention indicators without intervention
        protected_group_mask: Boolean mask for protected group
        
    Returns:
        EARG value
    """
    # Focus on protected group
    protected_retention_with = retention_with_intervention[protected_group_mask]
    protected_retention_without = retention_without_intervention[protected_group_mask]
    
    if len(protected_retention_with) == 0:
        return 0.0
    
    # Compute gain for protected group
    earg = (protected_retention_with.mean() - protected_retention_without.mean())
    
    return earg


def compute_cumulative_regret(
    rewards_sequence: List[np.ndarray],
    optimal_rewards: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    """
    Compute cumulative regret over time
    
    Args:
        rewards_sequence: List of reward arrays over time
        optimal_rewards: Optimal rewards (if None, use max observed)
        
    Returns:
        Cumulative regret array
    """
    if optimal_rewards is None:
        # Use maximum observed reward as optimal
        optimal_rewards = [r.max() * np.ones_like(r) for r in rewards_sequence]
    
    regrets = []
    cumulative = 0.0
    
    for actual, optimal in zip(rewards_sequence, optimal_rewards):
        instant_regret = (optimal - actual).mean()
        cumulative += instant_regret
        regrets.append(cumulative)
    
    return np.array(regrets)


def compute_demographic_parity(
    y_pred: np.ndarray,
    protected_attribute: np.ndarray
) -> float:
    """
    Compute demographic parity difference
    
    Args:
        y_pred: Predicted labels or probabilities
        protected_attribute: Protected group indicator
        
    Returns:
        Demographic parity difference
    """
    protected_rate = y_pred[protected_attribute == 1].mean()
    unprotected_rate = y_pred[protected_attribute == 0].mean()
    
    return abs(protected_rate - unprotected_rate)


def compute_equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attribute: np.ndarray
) -> Tuple[float, float]:
    """
    Compute equalized odds: difference in TPR and FPR between groups
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        protected_attribute: Protected group indicator
        
    Returns:
        Tuple of (TPR difference, FPR difference)
    """
    # Protected group
    protected_mask = protected_attribute == 1
    y_true_protected = y_true[protected_mask]
    y_pred_protected = y_pred[protected_mask]
    
    # Unprotected group
    unprotected_mask = protected_attribute == 0
    y_true_unprotected = y_true[unprotected_mask]
    y_pred_unprotected = y_pred[unprotected_mask]
    
    # Compute TPR and FPR for each group
    def tpr_fpr(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return tpr, fpr
    
    tpr_protected, fpr_protected = tpr_fpr(y_true_protected, y_pred_protected)
    tpr_unprotected, fpr_unprotected = tpr_fpr(y_true_unprotected, y_pred_unprotected)
    
    tpr_diff = abs(tpr_protected - tpr_unprotected)
    fpr_diff = abs(fpr_protected - fpr_unprotected)
    
    return tpr_diff, fpr_diff


def compute_all_metrics(
    dropout_probs: np.ndarray,
    dropout_labels: np.ndarray,
    trajectory_preds: np.ndarray,
    trajectory_labels: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    protected_groups: Optional[np.ndarray] = None,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics
    
    Args:
        dropout_probs: Predicted dropout probabilities
        dropout_labels: True dropout labels
        trajectory_preds: Predicted trajectory outcomes
        trajectory_labels: True trajectory outcomes
        actions: Intervention actions taken
        rewards: Rewards received
        protected_groups: Protected group indicators
        threshold: Classification threshold (if None, optimize)
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # === Predictive Metrics ===
    
    # AUC scores
    metrics['auc_pr'] = compute_auc_pr(dropout_labels, dropout_probs)
    metrics['auc_roc'] = compute_auc_roc(dropout_labels, dropout_probs)
    
    # Find optimal threshold if not provided
    if threshold is None:
        threshold, _ = find_optimal_threshold(dropout_labels, dropout_probs, metric='f2')
    
    # Binary predictions
    dropout_preds = (dropout_probs >= threshold).astype(int)
    
    # Classification metrics
    metrics['precision'] = precision_score(dropout_labels, dropout_preds, zero_division=0)
    metrics['recall'] = recall_score(dropout_labels, dropout_preds, zero_division=0)
    metrics['f1'] = f1_score(dropout_labels, dropout_preds, zero_division=0)
    metrics['f2'] = compute_f_beta_score(dropout_labels, dropout_preds, beta=2.0)
    
    # KS statistic
    metrics['ks_statistic'] = compute_ks_statistic(dropout_labels, dropout_probs)
    
    # === Policy Metrics ===
    
    # Retention rate (1 - dropout rate)
    metrics['retention_rate'] = (1 - dropout_labels).mean()
    
    # Mean reward
    metrics['mean_reward'] = rewards.mean()
    metrics['total_reward'] = rewards.sum()
    
    # Intervention rate
    metrics['intervention_rate'] = actions.mean()
    
    # === Trajectory Metrics ===
    
    # Trajectory gain
    metrics['trajectory_gain'] = compute_trajectory_gain(
        trajectory_preds.flatten(),
        trajectory_labels
    )
    
    # === Cost Metrics ===
    
    # Intervention efficiency ratio
    retention_gain = metrics['retention_rate']
    intervention_cost = actions.sum()  # Assuming unit cost
    metrics['ier'] = compute_intervention_efficiency_ratio(retention_gain, intervention_cost)
    
    # === Equity Metrics ===
    
    if protected_groups is not None:
        # EARG
        retention_with = (1 - dropout_labels).astype(float)
        retention_without = retention_with * (1 - actions)  # Simplified
        
        metrics['earg'] = compute_equity_adjusted_retention_gain(
            retention_with,
            retention_without,
            protected_groups.astype(bool)
        )
        
        # Demographic parity
        metrics['demographic_parity'] = compute_demographic_parity(
            actions,
            protected_groups
        )
        
        # Equalized odds
        tpr_diff, fpr_diff = compute_equalized_odds(
            dropout_labels,
            dropout_preds,
            protected_groups
        )
        metrics['equalized_odds_tpr'] = tpr_diff
        metrics['equalized_odds_fpr'] = fpr_diff
    
    return metrics


def print_metrics_table(metrics: Dict[str, float]):
    """
    Print metrics in a formatted table
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Group metrics by category
    predictive = ['auc_pr', 'auc_roc', 'precision', 'recall', 'f1', 'f2', 'ks_statistic']
    policy = ['retention_rate', 'mean_reward', 'intervention_rate']
    trajectory = ['trajectory_gain']
    cost = ['ier']
    equity = ['earg', 'demographic_parity', 'equalized_odds_tpr', 'equalized_odds_fpr']
    
    categories = [
        ("Predictive Metrics", predictive),
        ("Policy Metrics", policy),
        ("Trajectory Metrics", trajectory),
        ("Cost Metrics", cost),
        ("Equity Metrics", equity)
    ]
    
    for category_name, metric_names in categories:
        print(f"\n{category_name}:")
        print("-" * 60)
        for name in metric_names:
            if name in metrics:
                value = metrics[name]
                print(f"  {name:30s}: {value:8.4f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Test metrics
    print("Testing Evaluation Metrics...")
    
    np.random.seed(42)
    n = 1000
    
    # Generate synthetic data
    dropout_probs = np.random.rand(n)
    dropout_labels = (dropout_probs > 0.6).astype(int)
    trajectory_preds = np.random.randn(n)
    trajectory_labels = np.random.randn(n)
    actions = (np.random.rand(n) > 0.9).astype(int)
    rewards = np.random.randn(n)
    protected_groups = (np.random.rand(n) > 0.7).astype(int)
    
    # Compute all metrics
    metrics = compute_all_metrics(
        dropout_probs,
        dropout_labels,
        trajectory_preds,
        trajectory_labels,
        actions,
        rewards,
        protected_groups
    )
    
    # Print results
    print_metrics_table(metrics)
