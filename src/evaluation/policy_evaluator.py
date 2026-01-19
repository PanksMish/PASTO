"""
Policy-specific evaluation for PASTO
Evaluates sequential decision-making quality
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class PolicyEvaluator:
    """
    Evaluates policy performance over sequential decisions
    """
    
    def __init__(self, gamma: float = 0.95):
        """
        Initialize policy evaluator
        
        Args:
            gamma: Discount factor
        """
        self.gamma = gamma
        self.history = defaultdict(list)
        
    def evaluate_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        budget_ratio: float = 0.1
    ) -> Dict[str, float]:
        """
        Evaluate a single episode
        
        Args:
            states: State sequence (T, N, D)
            actions: Action sequence (T, N)
            rewards: Reward sequence (T, N)
            budget_ratio: Budget constraint
            
        Returns:
            Episode metrics
        """
        T, N = actions.shape
        
        # Compute discounted return
        discounted_return = 0.0
        for t in range(T):
            discounted_return += (self.gamma ** t) * rewards[t].mean()
        
        # Compute budget utilization
        budget_used = actions.sum() / (T * N * budget_ratio)
        
        # Compute intervention efficiency
        intervened_mask = actions.sum(axis=0) > 0
        avg_reward_intervened = rewards[:, intervened_mask].mean()
        avg_reward_not_intervened = rewards[:, ~intervened_mask].mean()
        intervention_lift = avg_reward_intervened - avg_reward_not_intervened
        
        metrics = {
            'discounted_return': discounted_return,
            'average_reward': rewards.mean(),
            'budget_utilization': budget_used,
            'intervention_lift': intervention_lift,
            'intervention_rate': actions.mean()
        }
        
        return metrics
    
    def compute_regret(
        self,
        actual_rewards: np.ndarray,
        optimal_rewards: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute cumulative regret
        
        Args:
            actual_rewards: Actual rewards received (T,)
            optimal_rewards: Optimal rewards (T,)
            
        Returns:
            Tuple of (total_regret, cumulative_regret_sequence)
        """
        instant_regret = optimal_rewards - actual_rewards
        cumulative_regret = np.cumsum(instant_regret)
        
        return cumulative_regret[-1], cumulative_regret


if __name__ == "__main__":
    # Test policy evaluator
    evaluator = PolicyEvaluator(gamma=0.95)
    
    T, N = 30, 100
    states = np.random.randn(T, N, 256)
    actions = np.random.randint(0, 2, (T, N))
    rewards = np.random.randn(T, N)
    
    metrics = evaluator.evaluate_episode(states, actions, rewards)
    
    print("Policy Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
