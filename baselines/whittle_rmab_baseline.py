"""
Classical Whittle Index RMAB Baseline
Fixed state space with computed Whittle indices
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


class WhittleRMABBaseline:
    """
    Classical Whittle Index RMAB with fixed state space
    """
    
    def __init__(
        self,
        num_states: int = 24,
        config: Dict = None
    ):
        """
        Initialize Whittle RMAB baseline
        
        Args:
            num_states: Number of discrete states
            config: Configuration dictionary
        """
        self.num_states = num_states
        self.config = config or {}
        
        # RMAB parameters
        self.gamma = self.config.get('gamma', 0.95)
        
        # State transition matrices (to be learned from data)
        # P[a][s, s'] = probability of transitioning from s to s' under action a
        self.transition_matrices = {
            0: None,  # Passive action
            1: None   # Active action (intervention)
        }
        
        # Reward function R[a][s]
        self.rewards = {
            0: None,
            1: None
        }
        
        # Whittle indices W[s]
        self.whittle_indices = None
        
    def learn_from_data(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray
    ):
        """
        Learn transition dynamics and rewards from data
        
        Args:
            states: Current states (N,)
            actions: Actions taken (N,)
            rewards: Rewards received (N,)
            next_states: Next states (N,)
        """
        print("Learning transition dynamics from data...")
        
        # Initialize transition matrices
        self.transition_matrices[0] = np.zeros((self.num_states, self.num_states))
        self.transition_matrices[1] = np.zeros((self.num_states, self.num_states))
        
        # Initialize reward arrays
        self.rewards[0] = np.zeros(self.num_states)
        self.rewards[1] = np.zeros(self.num_states)
        
        # Count transitions and accumulate rewards
        transition_counts = {0: np.zeros((self.num_states, self.num_states)),
                           1: np.zeros((self.num_states, self.num_states))}
        
        reward_counts = {0: np.zeros(self.num_states),
                        1: np.zeros(self.num_states)}
        
        reward_sums = {0: np.zeros(self.num_states),
                      1: np.zeros(self.num_states)}
        
        for s, a, r, s_next in zip(states, actions, rewards, next_states):
            s = int(s) % self.num_states
            s_next = int(s_next) % self.num_states
            a = int(a)
            
            transition_counts[a][s, s_next] += 1
            reward_sums[a][s] += r
            reward_counts[a][s] += 1
        
        # Normalize to get probabilities
        for a in [0, 1]:
            for s in range(self.num_states):
                total_count = transition_counts[a][s, :].sum()
                if total_count > 0:
                    self.transition_matrices[a][s, :] = \
                        transition_counts[a][s, :] / total_count
                else:
                    # Uniform distribution if no data
                    self.transition_matrices[a][s, :] = 1.0 / self.num_states
                
                # Average rewards
                if reward_counts[a][s] > 0:
                    self.rewards[a][s] = reward_sums[a][s] / reward_counts[a][s]
        
        print("✓ Transition dynamics learned")
        
    def compute_whittle_indices(self):
        """
        Compute Whittle indices for all states
        Uses value iteration with bisection search for subsidy
        """
        print("Computing Whittle indices...")
        
        self.whittle_indices = np.zeros(self.num_states)
        
        for s in range(self.num_states):
            # Find subsidy where active and passive actions have equal value
            def value_difference(subsidy):
                # Compute values with subsidy
                V_passive = self._compute_value_function(action=0, subsidy=0.0)
                V_active = self._compute_value_function(action=1, subsidy=subsidy)
                
                return V_active[s] - V_passive[s]
            
            # Bisection search for Whittle index
            try:
                result = minimize_scalar(
                    lambda w: abs(value_difference(w)),
                    bounds=(-10, 10),
                    method='bounded'
                )
                self.whittle_indices[s] = result.x
            except:
                self.whittle_indices[s] = 0.0
        
        print("✓ Whittle indices computed")
        
    def _compute_value_function(
        self,
        action: int,
        subsidy: float = 0.0,
        max_iters: int = 100,
        tol: float = 1e-4
    ) -> np.ndarray:
        """
        Compute value function for a fixed action using value iteration
        
        Args:
            action: Action to take (0 or 1)
            subsidy: Subsidy for active action
            max_iters: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Value function V[s]
        """
        V = np.zeros(self.num_states)
        
        for _ in range(max_iters):
            V_new = np.zeros(self.num_states)
            
            for s in range(self.num_states):
                # Reward (with subsidy if active)
                reward = self.rewards[action][s]
                if action == 1:
                    reward -= subsidy
                
                # Expected future value
                expected_value = np.dot(
                    self.transition_matrices[action][s, :],
                    V
                )
                
                V_new[s] = reward + self.gamma * expected_value
            
            # Check convergence
            if np.max(np.abs(V_new - V)) < tol:
                break
            
            V = V_new
        
        return V
    
    def discretize_states(self, features: np.ndarray) -> np.ndarray:
        """
        Simple discretization of continuous features to states
        
        Args:
            features: Feature vectors (N, F)
            
        Returns:
            Discrete states (N,)
        """
        # Simple quantile-based discretization
        # Compute principal component
        feature_mean = features.mean(axis=1)
        
        # Map to states using quantiles
        percentiles = np.linspace(0, 100, self.num_states + 1)
        bins = np.percentile(feature_mean, percentiles)
        
        states = np.digitize(feature_mean, bins) - 1
        states = np.clip(states, 0, self.num_states - 1)
        
        return states
    
    def select_actions(
        self,
        states: np.ndarray,
        budget_ratio: float = 0.1
    ) -> np.ndarray:
        """
        Select actions using Whittle index policy
        
        Args:
            states: Current states (N,)
            budget_ratio: Budget as fraction of population
            
        Returns:
            Actions (N,)
        """
        if self.whittle_indices is None:
            raise ValueError("Whittle indices not computed. Call compute_whittle_indices first.")
        
        N = len(states)
        num_interventions = max(1, int(N * budget_ratio))
        
        # Get Whittle indices for current states
        indices = self.whittle_indices[states.astype(int)]
        
        # Select top-k by Whittle index
        top_indices = np.argsort(indices)[-num_interventions:]
        
        actions = np.zeros(N, dtype=int)
        actions[top_indices] = 1
        
        return actions
    
    def fit(
        self,
        train_features: np.ndarray,
        train_actions: np.ndarray,
        train_rewards: np.ndarray
    ):
        """
        Fit Whittle RMAB from training data
        
        Args:
            train_features: Training features (N, T, F)
            train_actions: Training actions (N,)
            train_rewards: Training rewards (N,)
        """
        print("\n" + "="*60)
        print("FITTING WHITTLE RMAB BASELINE")
        print("="*60)
        
        # Flatten sequences and discretize states
        N, T, F = train_features.shape
        features_flat = train_features.reshape(N, T * F)
        
        states = self.discretize_states(features_flat)
        
        # Create next states (simplified: random walk)
        next_states = np.copy(states)
        for i in range(len(next_states)):
            if train_actions[i] == 0:  # Passive: tend to worsen
                next_states[i] = min(next_states[i] + np.random.randint(-1, 2), 
                                    self.num_states - 1)
            else:  # Active: tend to improve
                next_states[i] = max(next_states[i] - np.random.randint(0, 2), 0)
        
        # Learn dynamics
        self.learn_from_data(states, train_actions, train_rewards, next_states)
        
        # Compute Whittle indices
        self.compute_whittle_indices()
        
        print("\n✓ Whittle RMAB fitted successfully")
        print("="*60)
    
    def predict(
        self,
        features: np.ndarray,
        budget_ratio: float = 0.1
    ) -> np.ndarray:
        """
        Predict actions for new data
        
        Args:
            features: Feature sequences (N, T, F)
            budget_ratio: Budget ratio
            
        Returns:
            Actions (N,)
        """
        # Flatten and discretize
        N, T, F = features.shape
        features_flat = features.reshape(N, T * F)
        states = self.discretize_states(features_flat)
        
        # Select actions
        actions = self.select_actions(states, budget_ratio)
        
        return actions


if __name__ == "__main__":
    # Test Whittle RMAB baseline
    print("Testing Whittle RMAB Baseline...")
    
    N = 1000
    T, F = 30, 10
    
    # Create dummy data
    features = np.random.randn(N, T, F)
    actions = np.random.randint(0, 2, N)
    rewards = np.random.randn(N)
    
    # Initialize and fit
    model = WhittleRMABBaseline(num_states=24)
    model.fit(features, actions, rewards)
    
    # Test predictions
    test_features = np.random.randn(200, T, F)
    test_actions = model.predict(test_features, budget_ratio=0.1)
    
    print(f"\nTest actions shape: {test_actions.shape}")
    print(f"Interventions: {test_actions.sum()}")
    print(f"Intervention rate: {test_actions.mean():.2%}")
    
    # Print some Whittle indices
    print(f"\nWhittle indices (first 10 states):")
    print(model.whittle_indices[:10])
    
    print("\n✓ Whittle RMAB baseline test complete!")
