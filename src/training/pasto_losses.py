"""
Loss Functions for PASTO
Implements various loss functions for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DropoutLoss(nn.Module):
    """
    Binary cross-entropy loss for dropout prediction
    With optional class weighting for imbalanced data
    """
    
    def __init__(
        self,
        pos_weight: Optional[float] = None,
        label_smoothing: float = 0.0
    ):
        """
        Initialize dropout loss
        
        Args:
            pos_weight: Weight for positive class (dropout)
            label_smoothing: Label smoothing factor
        """
        super(DropoutLoss, self).__init__()
        
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dropout prediction loss
        
        Args:
            logits: Predicted logits (batch_size, 1) or (batch_size,)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Loss value
        """
        # Ensure correct shapes
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        
        targets = targets.float()
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Compute BCE loss
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor(self.pos_weight, device=logits.device)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=pos_weight_tensor
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        return loss


class TrajectoryLoss(nn.Module):
    """
    Loss for trajectory outcome prediction
    Supports MSE, MAE, and Huber loss
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        huber_delta: float = 1.0
    ):
        """
        Initialize trajectory loss
        
        Args:
            loss_type: Type of loss ('mse', 'mae', 'huber')
            huber_delta: Delta parameter for Huber loss
        """
        super(TrajectoryLoss, self).__init__()
        
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute trajectory prediction loss
        
        Args:
            predictions: Predicted trajectory outcomes (batch_size, 1) or (batch_size,)
            targets: Ground truth trajectories (batch_size, 1) or (batch_size,)
            
        Returns:
            Loss value
        """
        # Ensure same shape
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
        
        if self.loss_type == 'mse':
            loss = F.mse_loss(predictions, targets)
        elif self.loss_type == 'mae':
            loss = F.l1_loss(predictions, targets)
        elif self.loss_type == 'huber':
            loss = F.huber_loss(predictions, targets, delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class PolicyLoss(nn.Module):
    """
    Policy gradient loss for RMAB allocation
    """
    
    def __init__(
        self,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5
    ):
        """
        Initialize policy loss
        
        Args:
            entropy_coef: Coefficient for entropy regularization
            value_coef: Coefficient for value loss
        """
        super(PolicyLoss, self).__init__()
        
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
    def forward(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        target_values: Optional[torch.Tensor] = None,
        action_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute policy gradient loss
        
        Args:
            log_probs: Log probabilities of actions (batch_size,)
            advantages: Advantage estimates (batch_size,)
            values: Predicted values (batch_size,) - optional
            target_values: Target values (batch_size,) - optional
            action_probs: Action probabilities for entropy (batch_size,) - optional
            
        Returns:
            Total loss
        """
        # Policy gradient loss
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        total_loss = policy_loss
        
        # Add value loss if provided
        if values is not None and target_values is not None:
            value_loss = F.mse_loss(values, target_values.detach())
            total_loss = total_loss + self.value_coef * value_loss
        
        # Add entropy regularization if provided
        if action_probs is not None and self.entropy_coef > 0:
            # Binary entropy: -p*log(p) - (1-p)*log(1-p)
            entropy = -(
                action_probs * torch.log(action_probs + 1e-8) +
                (1 - action_probs) * torch.log(1 - action_probs + 1e-8)
            ).mean()
            
            total_loss = total_loss - self.entropy_coef * entropy
        
        return total_loss


class CompositeLoss(nn.Module):
    """
    Composite loss combining all components
    """
    
    def __init__(
        self,
        lambda_dropout: float = 0.6,
        lambda_trajectory: float = 0.4,
        lambda_policy: float = 0.1,
        dropout_pos_weight: Optional[float] = None
    ):
        """
        Initialize composite loss
        
        Args:
            lambda_dropout: Weight for dropout loss
            lambda_trajectory: Weight for trajectory loss
            lambda_policy: Weight for policy loss
            dropout_pos_weight: Positive class weight for dropout
        """
        super(CompositeLoss, self).__init__()
        
        self.lambda_dropout = lambda_dropout
        self.lambda_trajectory = lambda_trajectory
        self.lambda_policy = lambda_policy
        
        self.dropout_loss = DropoutLoss(pos_weight=dropout_pos_weight)
        self.trajectory_loss = TrajectoryLoss(loss_type='mse')
        self.policy_loss = PolicyLoss()
        
    def forward(
        self,
        dropout_logits: torch.Tensor,
        dropout_labels: torch.Tensor,
        trajectory_preds: torch.Tensor,
        trajectory_labels: torch.Tensor,
        log_probs: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute composite loss
        
        Args:
            dropout_logits: Dropout prediction logits
            dropout_labels: Dropout ground truth
            trajectory_preds: Trajectory predictions
            trajectory_labels: Trajectory ground truth
            log_probs: Log probabilities for policy gradient
            advantages: Advantage estimates
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Predictive losses
        d_loss = self.dropout_loss(dropout_logits, dropout_labels)
        t_loss = self.trajectory_loss(trajectory_preds, trajectory_labels)
        
        total_loss = (
            self.lambda_dropout * d_loss +
            self.lambda_trajectory * t_loss
        )
        
        loss_dict = {
            'dropout_loss': d_loss,
            'trajectory_loss': t_loss
        }
        
        # Add policy loss if provided
        if log_probs is not None and advantages is not None:
            p_loss = self.policy_loss(log_probs, advantages)
            total_loss = total_loss + self.lambda_policy * p_loss
            loss_dict['policy_loss'] = p_loss
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance
    Focuses on hard examples
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize focal loss
        
        Args:
            alpha: Weighting factor for class balance
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            logits: Predicted logits (batch_size,)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Loss value
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        targets = targets.float()
        
        # Compute focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weights
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Combine
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative representations
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        margin: float = 1.0
    ):
        """
        Initialize contrastive loss
        
        Args:
            temperature: Temperature for scaling
            margin: Margin for negative pairs
        """
        super(ContrastiveLoss, self).__init__()
        
        self.temperature = temperature
        self.margin = margin
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embeddings: Feature embeddings (batch_size, embed_dim)
            labels: Labels for determining positive/negative pairs (batch_size,)
            
        Returns:
            Loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # Create label matrix (1 if same class, 0 otherwise)
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # Positive pairs loss (minimize distance)
        pos_loss = (label_matrix * dist_matrix.pow(2)).sum() / (label_matrix.sum() + 1e-8)
        
        # Negative pairs loss (maximize distance beyond margin)
        neg_loss = ((1 - label_matrix) * F.relu(self.margin - dist_matrix).pow(2)).sum()
        neg_loss = neg_loss / ((1 - label_matrix).sum() + 1e-8)
        
        return pos_loss + neg_loss


if __name__ == "__main__":
    # Test losses
    print("Testing Loss Functions...")
    
    batch_size = 32
    
    # Test Dropout Loss
    print("\n1. Dropout Loss")
    dropout_loss = DropoutLoss(pos_weight=2.0)
    logits = torch.randn(batch_size)
    targets = torch.randint(0, 2, (batch_size,))
    loss = dropout_loss(logits, targets)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test Trajectory Loss
    print("\n2. Trajectory Loss")
    traj_loss = TrajectoryLoss(loss_type='mse')
    preds = torch.randn(batch_size)
    targets = torch.randn(batch_size)
    loss = traj_loss(preds, targets)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test Policy Loss
    print("\n3. Policy Loss")
    policy_loss = PolicyLoss()
    log_probs = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    loss = policy_loss(log_probs, advantages)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test Focal Loss
    print("\n4. Focal Loss")
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal(logits, targets)
    print(f"  Loss: {loss.item():.4f}")
    
    print("\n✓ All loss functions tested successfully!")
