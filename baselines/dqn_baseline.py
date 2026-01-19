"""
Deep Q-Network (DQN) Baseline for Sequential Intervention
Offline RL approach for student dropout intervention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from collections import deque
import random


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for action-value estimation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_dims: List[int] = [256, 128, 64]
    ):
        """
        Initialize DQN network
        
        Args:
            state_dim: State dimension
            action_dim: Number of actions (2 for intervene/no-intervene)
            hidden_dims: Hidden layer dimensions
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get Q-values
        
        Args:
            state: State tensor (batch_size, state_dim)
            
        Returns:
            Q-values (batch_size, action_dim)
        """
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for DQN
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch from buffer"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNBaseline:
    """
    DQN baseline for sequential intervention
    """
    
    def __init__(
        self,
        state_dim: int,
        config: Dict = None,
        device: str = 'cuda'
    ):
        """
        Initialize DQN baseline
        
        Args:
            state_dim: State dimension
            config: Configuration dictionary
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = 2  # intervene or not
        self.device = device
        self.config = config or {}
        
        # Hyperparameters
        self.gamma = self.config.get('gamma', 0.95)
        self.epsilon = self.config.get('epsilon_start', 1.0)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.learning_rate = self.config.get('learning_rate', 1e-3)
        self.batch_size = self.config.get('batch_size', 64)
        self.target_update_freq = self.config.get('target_update_freq', 10)
        
        # Networks
        self.q_network = DQNNetwork(state_dim, self.action_dim).to(device)
        self.target_network = DQNNetwork(state_dim, self.action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate
        )
        
        # Replay buffer
        buffer_size = self.config.get('buffer_size', 10000)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # Training stats
        self.update_count = 0
        
    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            explore: Whether to explore
            
        Returns:
            Action (0 or 1)
        """
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def select_actions_batch(
        self,
        states: np.ndarray,
        budget_ratio: float = 0.1,
        explore: bool = False
    ) -> np.ndarray:
        """
        Select actions for batch under budget constraint
        
        Args:
            states: Batch of states (N, state_dim)
            budget_ratio: Budget as fraction of population
            explore: Whether to explore
            
        Returns:
            Actions (N,)
        """
        N = len(states)
        num_interventions = max(1, int(N * budget_ratio))
        
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            q_values = self.q_network(states_tensor)
            
            # Q-value for intervention action (action=1)
            intervention_values = q_values[:, 1].cpu().numpy()
        
        # Select top-k by Q-value
        top_indices = np.argsort(intervention_values)[-num_interventions:]
        
        actions = np.zeros(N, dtype=int)
        actions[top_indices] = 1
        
        return actions
    
    def update(self):
        """Update Q-network using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q(s, a)
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train_offline(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        num_epochs: int = 100
    ):
        """
        Train DQN offline on collected data
        
        Args:
            states: States (N, state_dim)
            actions: Actions (N,)
            rewards: Rewards (N,)
            next_states: Next states (N, state_dim)
            dones: Done flags (N,)
            num_epochs: Number of training epochs
        """
        print("Training DQN offline...")
        
        # Add all experiences to replay buffer
        for i in range(len(states)):
            self.replay_buffer.push(
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                dones[i]
            )
        
        # Train for multiple epochs
        losses = []
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Multiple updates per epoch
            for _ in range(len(states) // self.batch_size):
                loss = self.update()
                if loss is not None:
                    epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.4f}")
        
        print("✓ DQN training complete")
        
        return losses
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"✓ DQN saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"✓ DQN loaded from {path}")


if __name__ == "__main__":
    # Test DQN baseline
    print("Testing DQN Baseline...")
    
    state_dim = 256
    N = 1000
    
    # Create dummy data
    states = np.random.randn(N, state_dim)
    actions = np.random.randint(0, 2, N)
    rewards = np.random.randn(N)
    next_states = np.random.randn(N, state_dim)
    dones = np.random.randint(0, 2, N)
    
    # Initialize DQN
    dqn = DQNBaseline(state_dim=state_dim, device='cpu')
    
    # Train offline
    losses = dqn.train_offline(
        states, actions, rewards, next_states, dones,
        num_epochs=10
    )
    
    # Test action selection
    test_states = np.random.randn(100, state_dim)
    actions = dqn.select_actions_batch(test_states, budget_ratio=0.1)
    
    print(f"\nTest actions shape: {actions.shape}")
    print(f"Interventions: {actions.sum()}")
    
    print("\n✓ DQN baseline test complete!")
