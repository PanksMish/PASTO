"""
PASTO: Policy-Aware Sequential Trajectory Optimization
Main model integrating all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

from .trajectory_encoder import TrajectoryEncoder, StateDiscretizer
from .dropout_predictor import DropoutRiskPredictor, TrajectoryOutcomePredictor, MultiTaskPredictor
from .whittle_index import (
    WhittleIndexNetwork, 
    ValueNetwork, 
    PolicyNetwork,
    IncentiveAlignedPrioritizer,
    AdvantageEstimator
)


class PASTO(nn.Module):
    """
    Complete PASTO framework integrating:
    - Module I: Predictive state modeling (trajectory encoding + prediction)
    - Module II: Prescriptive policy optimization (RMAB-based allocation)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize PASTO model
        
        Args:
            config: Configuration dictionary
        """
        super(PASTO, self).__init__()
        
        self.config = config
        model_config = config['model']
        policy_config = config['policy']
        
        # ===== Module I: Predictive State Modeling =====
        
        # 1. Trajectory Encoder
        self.trajectory_encoder = TrajectoryEncoder(config)
        encoder_output_dim = self.trajectory_encoder.output_dim
        
        # 2. State Discretizer
        state_config = model_config['state_space']
        self.state_discretizer = StateDiscretizer(
            input_dim=encoder_output_dim,
            num_risk_bins=state_config['num_risk_bins'],
            num_engagement_bins=state_config['num_engagement_bins'],
            num_transient_states=state_config['num_transient_states'],
            dropout_state_id=state_config['dropout_state_id']
        )
        
        # 3. Predictors
        if config.get('use_multitask_predictor', True):
            # Multi-task predictor (shares representations)
            self.predictor = MultiTaskPredictor(
                input_dim=encoder_output_dim,
                shared_hidden_dims=[256, 128],
                dropout_hidden_dims=[64],
                trajectory_hidden_dims=[64],
                dropout=0.2
            )
            self.use_multitask = True
        else:
            # Separate predictors
            dropout_config = model_config['dropout_predictor']
            self.dropout_predictor = DropoutRiskPredictor(
                input_dim=encoder_output_dim,
                hidden_dims=dropout_config['hidden_dims'],
                dropout=dropout_config['dropout'],
                activation=dropout_config['activation'],
                use_batch_norm=dropout_config['use_batch_norm']
            )
            
            trajectory_config = model_config['trajectory_predictor']
            self.trajectory_predictor = TrajectoryOutcomePredictor(
                input_dim=encoder_output_dim,
                hidden_dims=trajectory_config['hidden_dims'],
                dropout=trajectory_config['dropout'],
                activation=trajectory_config['activation'],
                output_dim=trajectory_config['output_dim']
            )
            self.use_multitask = False
        
        # ===== Module II: Prescriptive Policy Optimization =====
        
        # 4. Whittle Index Network
        whittle_config = model_config['whittle_index']
        self.whittle_index_net = WhittleIndexNetwork(
            state_dim=encoder_output_dim,
            hidden_dims=whittle_config['hidden_dims'],
            dropout=whittle_config['dropout'],
            activation=whittle_config['activation'],
            use_layer_norm=whittle_config['use_layer_norm'],
            output_activation=whittle_config['output_activation']
        )
        
        # 5. Value Network (for advantage estimation)
        critic_config = config['training']['critic']
        self.value_net = ValueNetwork(
            state_dim=encoder_output_dim,
            hidden_dims=critic_config['hidden_dims'],
            dropout=critic_config['dropout']
        )
        
        # 6. Policy Network
        self.policy_net = PolicyNetwork(
            whittle_index_net=self.whittle_index_net,
            initial_threshold=0.0
        )
        
        # 7. Incentive-Aligned Prioritizer
        if policy_config['use_incentive_alignment']:
            self.prioritizer = IncentiveAlignedPrioritizer(
                beta1_init=policy_config['beta1_init'],
                beta2_init=policy_config['beta2_init'],
                monotone_constraint=policy_config['monotone_constraint']
            )
        else:
            self.prioritizer = None
        
        # 8. Advantage Estimator
        self.advantage_estimator = AdvantageEstimator(
            gamma=policy_config['gamma'],
            gae_lambda=policy_config.get('gae_lambda', 0.95),
            method=policy_config.get('advantage_method', 'td')
        )
        
        # Policy parameters
        self.alpha = policy_config['alpha']  # Retention vs trajectory trade-off
        self.gamma = policy_config['gamma']  # Discount factor
        
    def forward(
        self,
        sequences: torch.Tensor,
        return_all: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PASTO
        
        Args:
            sequences: Input sequences (batch_size, seq_len, input_dim)
            return_all: Whether to return all outputs
            
        Returns:
            Dictionary containing all predictions and representations
        """
        batch_size = sequences.size(0)
        
        # Module I: Trajectory Encoding
        seq_outputs, final_hidden = self.trajectory_encoder(sequences)
        
        # Predictions
        if self.use_multitask:
            dropout_logits, trajectory_pred = self.predictor(final_hidden)
        else:
            dropout_logits = self.dropout_predictor(final_hidden)
            trajectory_pred = self.trajectory_predictor(final_hidden)
        
        # Dropout probabilities
        dropout_probs = torch.sigmoid(dropout_logits).squeeze(-1)
        
        # Discrete states
        discrete_states = self.state_discretizer(final_hidden)
        
        # Continuous risk and engagement scores
        risk_scores, engagement_scores = self.state_discretizer.get_continuous_scores(
            final_hidden
        )
        
        # Module II: Policy Optimization
        
        # Whittle indices
        whittle_indices = self.whittle_index_net(final_hidden).squeeze(-1)
        
        # State values
        state_values = self.value_net(final_hidden).squeeze(-1)
        
        # Action probabilities
        action_probs = self.policy_net(final_hidden)
        
        # Priority scores (for allocation)
        if self.prioritizer is not None:
            priority_scores = self.prioritizer(dropout_probs, whittle_indices)
        else:
            priority_scores = whittle_indices
        
        outputs = {
            'final_hidden': final_hidden,
            'dropout_logits': dropout_logits,
            'dropout_probs': dropout_probs,
            'trajectory_pred': trajectory_pred,
            'discrete_states': discrete_states,
            'risk_scores': risk_scores,
            'engagement_scores': engagement_scores,
            'whittle_indices': whittle_indices,
            'state_values': state_values,
            'action_probs': action_probs,
            'priority_scores': priority_scores
        }
        
        if return_all:
            outputs['seq_outputs'] = seq_outputs
        
        return outputs
    
    def compute_reward(
        self,
        dropout_labels: torch.Tensor,
        trajectory_labels: torch.Tensor,
        trajectory_pred: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rewards based on retention and trajectory improvement
        
        Args:
            dropout_labels: Binary dropout labels (batch_size,)
            trajectory_labels: Ground truth trajectories (batch_size,)
            trajectory_pred: Predicted trajectories (batch_size,)
            actions: Intervention actions (batch_size,)
            
        Returns:
            Rewards (batch_size,)
        """
        # Retention reward (binary: 1 if retained, 0 if dropped out)
        retention_reward = (1 - dropout_labels).float()
        
        # Trajectory improvement (normalized)
        trajectory_improvement = trajectory_pred.squeeze() - trajectory_labels
        trajectory_improvement = torch.tanh(trajectory_improvement)  # Normalize
        
        # Combined reward
        reward = (
            self.alpha * retention_reward + 
            (1 - self.alpha) * trajectory_improvement
        )
        
        # Add intervention cost (negative for intervened students)
        intervention_cost = actions * 0.1  # Small cost
        reward = reward - intervention_cost
        
        return reward
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        state_values: torch.Tensor,
        next_state_values: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute advantage estimates
        
        Args:
            rewards: Rewards (batch_size,)
            state_values: Current state values (batch_size,)
            next_state_values: Next state values (batch_size,)
            dones: Episode done flags (batch_size,)
            
        Returns:
            Advantages (batch_size,)
        """
        return self.advantage_estimator.compute_advantages(
            rewards, state_values, next_state_values, dones
        )
    
    def sample_actions(
        self,
        sequences: torch.Tensor,
        budget_ratio: float = 0.1,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample intervention actions under budget constraint
        
        Args:
            sequences: Input sequences (batch_size, seq_len, input_dim)
            budget_ratio: Budget as fraction of population
            deterministic: Use greedy selection
            
        Returns:
            Tuple of (actions, priority_scores, outputs_dict)
        """
        # Forward pass
        outputs = self.forward(sequences, return_all=False)
        
        priority_scores = outputs['priority_scores']
        batch_size = len(priority_scores)
        
        # Determine budget
        num_interventions = max(1, int(batch_size * budget_ratio))
        
        if deterministic:
            # Greedy: select top-k by priority
            _, top_indices = torch.topk(priority_scores, num_interventions)
            actions = torch.zeros(batch_size, device=sequences.device)
            actions[top_indices] = 1.0
        else:
            # Stochastic: sample based on probabilities
            action_probs = outputs['action_probs']
            
            # Sample actions
            dist = torch.distributions.Bernoulli(action_probs)
            actions = dist.sample()
            
            # Ensure budget constraint (select top-k if exceeded)
            num_selected = actions.sum().item()
            if num_selected > num_interventions:
                # Re-select top-k by priority among sampled
                sampled_priorities = priority_scores * actions
                _, top_indices = torch.topk(sampled_priorities, num_interventions)
                actions = torch.zeros(batch_size, device=sequences.device)
                actions[top_indices] = 1.0
        
        return actions, priority_scores, outputs
    
    def get_encoder_parameters(self):
        """Get trajectory encoder parameters"""
        params = list(self.trajectory_encoder.parameters())
        if self.use_multitask:
            params += list(self.predictor.parameters())
        else:
            params += list(self.dropout_predictor.parameters())
            params += list(self.trajectory_predictor.parameters())
        params += list(self.state_discretizer.parameters())
        return params
    
    def get_policy_parameters(self):
        """Get policy network parameters"""
        params = list(self.whittle_index_net.parameters())
        params += list(self.policy_net.parameters())
        if self.prioritizer is not None:
            params += list(self.prioritizer.parameters())
        return params
    
    def get_value_parameters(self):
        """Get value network parameters"""
        return list(self.value_net.parameters())


class PASTOSimulator:
    """
    Simulator for evaluating PASTO in sequential intervention scenarios
    """
    
    def __init__(
        self,
        model: PASTO,
        config: Dict,
        device: str = 'cuda'
    ):
        """
        Initialize PASTO simulator
        
        Args:
            model: PASTO model
            config: Configuration dictionary
            device: Device to run on
        """
        self.model = model
        self.config = config
        self.device = device
        
        sim_config = config.get('simulation', {})
        self.num_students = sim_config.get('num_students', 1000)
        self.num_episodes = sim_config.get('num_episodes', 30)
        self.intervention_effect = sim_config.get('intervention_effect', {})
        
    def simulate_episode(
        self,
        sequences: torch.Tensor,
        dropout_labels: torch.Tensor,
        trajectory_labels: torch.Tensor,
        budget_ratio: float = 0.1
    ) -> Dict[str, List]:
        """
        Simulate a full episode of sequential interventions
        
        Args:
            sequences: Student sequences (N, T, F)
            dropout_labels: Dropout labels (N,)
            trajectory_labels: Trajectory labels (N,)
            budget_ratio: Intervention budget ratio
            
        Returns:
            Dictionary with simulation results
        """
        self.model.eval()
        
        results = {
            'actions': [],
            'rewards': [],
            'dropout_probs': [],
            'retention_rate': [],
            'cumulative_reward': []
        }
        
        cumulative_reward = 0
        
        with torch.no_grad():
            for t in range(self.num_episodes):
                # Sample actions
                actions, priorities, outputs = self.model.sample_actions(
                    sequences,
                    budget_ratio=budget_ratio,
                    deterministic=True
                )
                
                # Compute rewards
                rewards = self.model.compute_reward(
                    dropout_labels,
                    trajectory_labels,
                    outputs['trajectory_pred'],
                    actions
                )
                
                # Update cumulative reward
                cumulative_reward += rewards.sum().item()
                
                # Record results
                results['actions'].append(actions.cpu().numpy())
                results['rewards'].append(rewards.cpu().numpy())
                results['dropout_probs'].append(outputs['dropout_probs'].cpu().numpy())
                results['retention_rate'].append((1 - dropout_labels.float()).mean().item())
                results['cumulative_reward'].append(cumulative_reward)
        
        return results


if __name__ == "__main__":
    # Test PASTO model
    print("Testing PASTO Model...")
    
    # Create dummy config
    config = {
        'model': {
            'encoder': {
                'type': 'lstm',
                'input_dim': 64,
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.3,
                'bidirectional': True
            },
            'state_space': {
                'num_risk_bins': 4,
                'num_engagement_bins': 4,
                'num_transient_states': 7,
                'dropout_state_id': 24,
                'total_states': 24
            },
            'dropout_predictor': {
                'hidden_dims': [256, 128, 64],
                'dropout': 0.2,
                'activation': 'relu',
                'use_batch_norm': True
            },
            'trajectory_predictor': {
                'hidden_dims': [256, 128],
                'dropout': 0.2,
                'activation': 'relu',
                'output_dim': 1
            },
            'whittle_index': {
                'hidden_dims': [128, 64, 32],
                'dropout': 0.1,
                'activation': 'relu',
                'use_layer_norm': True,
                'output_activation': 'linear'
            }
        },
        'policy': {
            'alpha': 0.7,
            'gamma': 0.95,
            'use_incentive_alignment': True,
            'beta1_init': 0.6,
            'beta2_init': 0.4,
            'monotone_constraint': True,
            'gae_lambda': 0.95,
            'advantage_method': 'td'
        },
        'training': {
            'critic': {
                'hidden_dims': [128, 64],
                'dropout': 0.1
            }
        }
    }
    
    # Initialize model
    model = PASTO(config)
    
    # Test forward pass
    batch_size = 32
    seq_len = 30
    input_dim = 64
    
    sequences = torch.randn(batch_size, seq_len, input_dim)
    
    outputs = model(sequences)
    
    print("\nModel Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test action sampling
    print("\nTesting Action Sampling...")
    actions, priorities, outputs = model.sample_actions(sequences, budget_ratio=0.1)
    
    print(f"  Actions shape: {actions.shape}")
    print(f"  Number of interventions: {actions.sum().item()}")
    print(f"  Priorities shape: {priorities.shape}")
    
    # Test reward computation
    print("\nTesting Reward Computation...")
    dropout_labels = torch.randint(0, 2, (batch_size,))
    trajectory_labels = torch.randn(batch_size)
    
    rewards = model.compute_reward(
        dropout_labels,
        trajectory_labels,
        outputs['trajectory_pred'],
        actions
    )
    
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Mean reward: {rewards.mean():.3f}")
    
    print("\n✓ PASTO model test complete!")
