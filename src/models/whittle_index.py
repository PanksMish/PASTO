"""
Whittle Index Module for PASTO
Implements parameterized Whittle index learning for RMAB-based allocation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class WhittleIndexNetwork(nn.Module):
    """
    Parameterized Whittle Index network
    Learns to compute indices for sequential intervention allocation
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.1,
        activation: str = 'relu',
        use_layer_norm: bool = True,
        output_activation: str = 'linear'
    ):
        """
        Initialize Whittle Index network
        
        Args:
            state_dim: Dimension of state representation
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function
            use_layer_norm: Whether to use layer normalization
            output_activation: Output activation ('linear', 'softplus', 'sigmoid')
        """
        super(WhittleIndexNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.use_layer_norm = use_layer_norm
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Layer normalization
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        # Output activation
        if output_activation == 'softplus':
            layers.append(nn.Softplus())
        elif output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        # 'linear' means no activation
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Whittle indices for given states
        
        Args:
            state: State representations (batch_size, state_dim)
            
        Returns:
            Whittle indices (batch_size, 1)
        """
        indices = self.network(state)
        return indices


class ValueNetwork(nn.Module):
    """
    Value function network for advantage estimation
    Estimates V(s) for computing advantages A(s,a) = Q(s,a) - V(s)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: list = [128, 64],
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Initialize value network
        
        Args:
            state_dim: Dimension of state representation
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function
        """
        super(ValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Select activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute state values
        
        Args:
            state: State representations (batch_size, state_dim)
            
        Returns:
            State values (batch_size, 1)
        """
        values = self.network(state)
        return values


class AdvantageEstimator:
    """
    Computes advantage estimates for policy gradient methods
    Supports TD(0) and GAE (Generalized Advantage Estimation)
    """
    
    def __init__(
        self,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        method: str = 'td'
    ):
        """
        Initialize advantage estimator
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            method: Estimation method ('td' or 'gae')
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.method = method
        
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute advantages using specified method
        
        Args:
            rewards: Rewards (batch_size,)
            values: State values V(s_t) (batch_size,)
            next_values: Next state values V(s_{t+1}) (batch_size,)
            dones: Done flags (batch_size,)
            
        Returns:
            Advantages (batch_size,)
        """
        if dones is None:
            dones = torch.zeros_like(rewards)
        
        if self.method == 'td':
            return self._compute_td_advantages(rewards, values, next_values, dones)
        elif self.method == 'gae':
            return self._compute_gae_advantages(rewards, values, next_values, dones)
        else:
            raise ValueError(f"Unknown advantage method: {self.method}")
    
    def _compute_td_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute TD(0) advantages: A = r + γV(s') - V(s)"""
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        return advantages
    
    def _compute_gae_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        # Compute TD errors
        td_errors = rewards + self.gamma * next_values * (1 - dones) - values
        
        # Compute GAE
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # Backward pass to compute GAE
        for t in reversed(range(len(rewards))):
            delta = td_errors[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages


class PolicyNetwork(nn.Module):
    """
    Stochastic policy network for intervention selection
    π(a|s) = σ(W(s) - τ)
    """
    
    def __init__(
        self,
        whittle_index_net: WhittleIndexNetwork,
        initial_threshold: float = 0.0
    ):
        """
        Initialize policy network
        
        Args:
            whittle_index_net: Whittle index network
            initial_threshold: Initial threshold value
        """
        super(PolicyNetwork, self).__init__()
        
        self.whittle_index_net = whittle_index_net
        
        # Learnable threshold parameter
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))
        
    def forward(
        self,
        state: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute action probabilities
        
        Args:
            state: State representations (batch_size, state_dim)
            temperature: Temperature for softmax (higher = more exploration)
            
        Returns:
            Action probabilities (batch_size,)
        """
        # Compute Whittle indices
        indices = self.whittle_index_net(state).squeeze(-1)
        
        # Apply threshold with temperature
        logits = (indices - self.threshold) / temperature
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        return probs
    
    def sample_actions(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from policy
        
        Args:
            state: State representations (batch_size, state_dim)
            deterministic: If True, use greedy selection
            
        Returns:
            Tuple of (actions, log_probs)
        """
        probs = self.forward(state)
        
        if deterministic:
            # Greedy: select actions with prob > 0.5
            actions = (probs > 0.5).float()
            log_probs = torch.log(probs + 1e-8)
        else:
            # Stochastic sampling
            dist = torch.distributions.Bernoulli(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        
        return actions, log_probs


class IncentiveAlignedPrioritizer(nn.Module):
    """
    Monotone scoring function for incentive-aligned student prioritization
    φ(s) = β₁·p_drop + β₂·W(s)
    """
    
    def __init__(
        self,
        beta1_init: float = 0.6,
        beta2_init: float = 0.4,
        monotone_constraint: bool = True
    ):
        """
        Initialize incentive-aligned prioritizer
        
        Args:
            beta1_init: Initial weight for dropout risk
            beta2_init: Initial weight for Whittle index
            monotone_constraint: Enforce non-negativity of weights
        """
        super(IncentiveAlignedPrioritizer, self).__init__()
        
        self.monotone_constraint = monotone_constraint
        
        # Learnable weights
        if monotone_constraint:
            # Use softplus to ensure non-negativity
            self.beta1_raw = nn.Parameter(torch.tensor(np.log(np.exp(beta1_init) - 1)))
            self.beta2_raw = nn.Parameter(torch.tensor(np.log(np.exp(beta2_init) - 1)))
        else:
            self.beta1 = nn.Parameter(torch.tensor(beta1_init))
            self.beta2 = nn.Parameter(torch.tensor(beta2_init))
    
    def get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current weight values"""
        if self.monotone_constraint:
            beta1 = F.softplus(self.beta1_raw)
            beta2 = F.softplus(self.beta2_raw)
        else:
            beta1 = self.beta1
            beta2 = self.beta2
        
        return beta1, beta2
    
    def forward(
        self,
        dropout_risk: torch.Tensor,
        whittle_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute priority scores
        
        Args:
            dropout_risk: Dropout risk scores (batch_size,)
            whittle_indices: Whittle indices (batch_size,)
            
        Returns:
            Priority scores (batch_size,)
        """
        beta1, beta2 = self.get_weights()
        
        # Compute composite priority
        priority = beta1 * dropout_risk + beta2 * whittle_indices
        
        return priority


class BudgetConstrainedAllocator:
    """
    Greedy allocator under budget constraints
    Selects top-k students based on priority scores
    """
    
    def __init__(
        self,
        budget_ratio: float = 0.1,
        intervention_cost: float = 1.0
    ):
        """
        Initialize budget-constrained allocator
        
        Args:
            budget_ratio: Fraction of students to intervene on
            intervention_cost: Cost per intervention
        """
        self.budget_ratio = budget_ratio
        self.intervention_cost = intervention_cost
        
    def allocate(
        self,
        priority_scores: torch.Tensor,
        budget: Optional[float] = None
    ) -> torch.Tensor:
        """
        Allocate interventions under budget constraint
        
        Args:
            priority_scores: Priority scores (batch_size,)
            budget: Total budget (if None, use budget_ratio)
            
        Returns:
            Binary allocation decisions (batch_size,)
        """
        batch_size = len(priority_scores)
        
        # Determine number of interventions
        if budget is None:
            num_interventions = int(batch_size * self.budget_ratio)
        else:
            num_interventions = int(budget / self.intervention_cost)
        
        # Ensure at least one intervention
        num_interventions = max(1, min(num_interventions, batch_size))
        
        # Select top-k by priority
        _, top_indices = torch.topk(priority_scores, num_interventions)
        
        # Create allocation tensor
        allocations = torch.zeros_like(priority_scores)
        allocations[top_indices] = 1.0
        
        return allocations


if __name__ == "__main__":
    # Test Whittle Index Network
    print("Testing Whittle Index Network...")
    whittle_net = WhittleIndexNetwork(
        state_dim=256,
        hidden_dims=[128, 64, 32],
        dropout=0.1
    )
    
    states = torch.randn(32, 256)
    indices = whittle_net(states)
    
    print(f"  State shape: {states.shape}")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Index range: [{indices.min():.3f}, {indices.max():.3f}]")
    
    # Test Value Network
    print("\nTesting Value Network...")
    value_net = ValueNetwork(
        state_dim=256,
        hidden_dims=[128, 64]
    )
    
    values = value_net(states)
    print(f"  Values shape: {values.shape}")
    
    # Test Advantage Estimator
    print("\nTesting Advantage Estimator...")
    estimator = AdvantageEstimator(gamma=0.95, method='td')
    
    rewards = torch.randn(32)
    values = torch.randn(32)
    next_values = torch.randn(32)
    
    advantages = estimator.compute_advantages(rewards, values, next_values)
    print(f"  Advantages shape: {advantages.shape}")
    print(f"  Mean advantage: {advantages.mean():.3f}")
    
    # Test Policy Network
    print("\nTesting Policy Network...")
    policy = PolicyNetwork(whittle_net)
    
    probs = policy(states)
    actions, log_probs = policy.sample_actions(states)
    
    print(f"  Action probabilities shape: {probs.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Mean action probability: {probs.mean():.3f}")
    
    # Test Incentive-Aligned Prioritizer
    print("\nTesting Incentive-Aligned Prioritizer...")
    prioritizer = IncentiveAlignedPrioritizer()
    
    dropout_risk = torch.rand(32)
    priority = prioritizer(dropout_risk, indices.squeeze())
    
    print(f"  Priority scores shape: {priority.shape}")
    beta1, beta2 = prioritizer.get_weights()
    print(f"  β₁ = {beta1:.3f}, β₂ = {beta2:.3f}")
    
    # Test Budget-Constrained Allocator
    print("\nTesting Budget-Constrained Allocator...")
    allocator = BudgetConstrainedAllocator(budget_ratio=0.1)
    
    allocations = allocator.allocate(priority)
    print(f"  Allocations shape: {allocations.shape}")
    print(f"  Number allocated: {allocations.sum().item()}")
    print(f"  Allocation rate: {allocations.mean():.2%}")
