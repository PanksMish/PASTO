"""
Dropout Predictor Module for PASTO
Predicts short-term dropout risk and long-term trajectory outcomes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class DropoutRiskPredictor(nn.Module):
    """
    Predicts dropout risk from latent trajectory representations
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        activation: str = 'relu',
        use_batch_norm: bool = True
    ):
        """
        Initialize dropout risk predictor
        
        Args:
            input_dim: Input latent dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
            use_batch_norm: Whether to use batch normalization
        """
        super(DropoutRiskPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict dropout risk
        
        Args:
            h: Latent representations (batch_size, input_dim)
            
        Returns:
            Dropout risk logits (batch_size, 1)
        """
        logits = self.network(h)
        return logits
    
    def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict dropout probabilities
        
        Args:
            h: Latent representations (batch_size, input_dim)
            
        Returns:
            Dropout probabilities (batch_size,)
        """
        logits = self.forward(h)
        probs = torch.sigmoid(logits).squeeze(-1)
        return probs


class TrajectoryOutcomePredictor(nn.Module):
    """
    Predicts long-term trajectory outcomes from temporal representations
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
        activation: str = 'relu',
        output_dim: int = 1
    ):
        """
        Initialize trajectory outcome predictor
        
        Args:
            input_dim: Input latent dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function
            output_dim: Output dimension (1 for regression)
        """
        super(TrajectoryOutcomePredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict trajectory outcome
        
        Args:
            h: Latent representations (batch_size, input_dim)
            
        Returns:
            Trajectory outcome predictions (batch_size, output_dim)
        """
        output = self.network(h)
        return output


class TemporalDropoutPredictor(nn.Module):
    """
    Predicts dropout risk at each time step using temporal features
    Enables early warning at different points in the trajectory
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize temporal dropout predictor
        
        Args:
            input_dim: Input feature dimension per time step
            hidden_dim: Hidden dimension
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super(TemporalDropoutPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_all_steps: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for temporal dropout prediction
        
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            return_all_steps: If True, return predictions for all time steps
            
        Returns:
            Dropout risk predictions
            - If return_all_steps=False: (batch_size, 1)
            - If return_all_steps=True: (batch_size, seq_len, 1)
        """
        # GRU encoding
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim)
        
        if return_all_steps:
            # Predict at each time step
            predictions = self.predictor(gru_out)  # (batch_size, seq_len, 1)
        else:
            # Predict only at final time step
            final_hidden = gru_out[:, -1, :]  # (batch_size, hidden_dim)
            predictions = self.predictor(final_hidden)  # (batch_size, 1)
        
        return predictions


class MultiTaskPredictor(nn.Module):
    """
    Multi-task predictor for both dropout and trajectory outcomes
    Shares representation learning across tasks
    """
    
    def __init__(
        self,
        input_dim: int,
        shared_hidden_dims: List[int] = [256, 128],
        dropout_hidden_dims: List[int] = [64],
        trajectory_hidden_dims: List[int] = [64],
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        """
        Initialize multi-task predictor
        
        Args:
            input_dim: Input latent dimension
            shared_hidden_dims: Shared network hidden dimensions
            dropout_hidden_dims: Dropout task-specific hidden dimensions
            trajectory_hidden_dims: Trajectory task-specific hidden dimensions
            dropout: Dropout probability
            activation: Activation function
        """
        super(MultiTaskPredictor, self).__init__()
        
        # Select activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared network
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in shared_hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(self.activation)
            shared_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        shared_output_dim = prev_dim
        
        # Dropout prediction head
        dropout_layers = []
        prev_dim = shared_output_dim
        
        for hidden_dim in dropout_hidden_dims:
            dropout_layers.append(nn.Linear(prev_dim, hidden_dim))
            dropout_layers.append(self.activation)
            dropout_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        dropout_layers.append(nn.Linear(prev_dim, 1))
        self.dropout_head = nn.Sequential(*dropout_layers)
        
        # Trajectory prediction head
        trajectory_layers = []
        prev_dim = shared_output_dim
        
        for hidden_dim in trajectory_hidden_dims:
            trajectory_layers.append(nn.Linear(prev_dim, hidden_dim))
            trajectory_layers.append(self.activation)
            trajectory_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        trajectory_layers.append(nn.Linear(prev_dim, 1))
        self.trajectory_head = nn.Sequential(*trajectory_layers)
        
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-task prediction
        
        Args:
            h: Latent representations (batch_size, input_dim)
            
        Returns:
            Tuple of (dropout_logits, trajectory_predictions)
        """
        # Shared representation
        shared_features = self.shared_network(h)
        
        # Task-specific predictions
        dropout_logits = self.dropout_head(shared_features)
        trajectory_pred = self.trajectory_head(shared_features)
        
        return dropout_logits, trajectory_pred


class EnsemblePredictor(nn.Module):
    """
    Ensemble of multiple predictors for robust predictions
    """
    
    def __init__(
        self,
        input_dim: int,
        num_models: int = 5,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2
    ):
        """
        Initialize ensemble predictor
        
        Args:
            input_dim: Input dimension
            num_models: Number of models in ensemble
            hidden_dims: Hidden dimensions for each model
            dropout: Dropout probability
        """
        super(EnsemblePredictor, self).__init__()
        
        self.num_models = num_models
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            DropoutRiskPredictor(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
            for _ in range(num_models)
        ])
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble
        
        Args:
            h: Latent representations (batch_size, input_dim)
            
        Returns:
            Averaged predictions (batch_size, 1)
        """
        predictions = []
        
        for model in self.models:
            pred = model(h)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)
        
        return ensemble_pred
    
    def predict_proba_with_uncertainty(
        self,
        h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimates
        
        Args:
            h: Latent representations (batch_size, input_dim)
            
        Returns:
            Tuple of (mean_probs, std_probs)
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict_proba(h)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (num_models, batch_size)
        
        mean_probs = predictions.mean(dim=0)
        std_probs = predictions.std(dim=0)
        
        return mean_probs, std_probs


if __name__ == "__main__":
    # Test Dropout Risk Predictor
    print("Testing Dropout Risk Predictor...")
    predictor = DropoutRiskPredictor(
        input_dim=256,
        hidden_dims=[256, 128, 64],
        dropout=0.2
    )
    
    h = torch.randn(32, 256)
    logits = predictor(h)
    probs = predictor.predict_proba(h)
    
    print(f"  Input shape: {h.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    # Test Trajectory Outcome Predictor
    print("\nTesting Trajectory Outcome Predictor...")
    traj_predictor = TrajectoryOutcomePredictor(
        input_dim=256,
        hidden_dims=[256, 128],
        dropout=0.2
    )
    
    traj_pred = traj_predictor(h)
    print(f"  Trajectory predictions shape: {traj_pred.shape}")
    
    # Test Multi-Task Predictor
    print("\nTesting Multi-Task Predictor...")
    multitask = MultiTaskPredictor(
        input_dim=256,
        shared_hidden_dims=[256, 128],
        dropout_hidden_dims=[64],
        trajectory_hidden_dims=[64]
    )
    
    dropout_logits, traj_pred = multitask(h)
    print(f"  Dropout logits shape: {dropout_logits.shape}")
    print(f"  Trajectory predictions shape: {traj_pred.shape}")
    
    # Test Ensemble Predictor
    print("\nTesting Ensemble Predictor...")
    ensemble = EnsemblePredictor(
        input_dim=256,
        num_models=5,
        hidden_dims=[256, 128, 64]
    )
    
    ensemble_pred = ensemble(h)
    mean_probs, std_probs = ensemble.predict_proba_with_uncertainty(h)
    
    print(f"  Ensemble predictions shape: {ensemble_pred.shape}")
    print(f"  Mean probabilities shape: {mean_probs.shape}")
    print(f"  Std probabilities shape: {std_probs.shape}")
    print(f"  Mean uncertainty: {std_probs.mean():.4f}")
