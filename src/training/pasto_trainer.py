"""
Training Module for PASTO
Implements the complete training loop with policy gradient optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, List
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import time

from ..models.pasto import PASTO
from .losses import CompositeLoss, DropoutLoss, TrajectoryLoss, PolicyLoss
from ..evaluation.metrics import compute_all_metrics


class PASTOTrainer:
    """
    Trainer for PASTO framework
    Handles joint optimization of trajectory encoder and policy network
    """
    
    def __init__(
        self,
        model: PASTO,
        config: Dict,
        device: str = 'cuda',
        save_dir: str = 'checkpoints'
    ):
        """
        Initialize PASTO trainer
        
        Args:
            model: PASTO model
            config: Configuration dictionary
            device: Device to train on
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        train_config = config['training']
        
        # Optimizers
        self.encoder_optimizer = self._create_optimizer(
            self.model.get_encoder_parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        self.policy_optimizer = self._create_optimizer(
            self.model.get_policy_parameters(),
            lr=train_config['policy_lr'],
            weight_decay=train_config['weight_decay']
        )
        
        self.value_optimizer = self._create_optimizer(
            self.model.get_value_parameters(),
            lr=train_config['value_lr'],
            weight_decay=train_config['weight_decay']
        )
        
        # Learning rate schedulers
        self.encoder_scheduler = self._create_scheduler(
            self.encoder_optimizer,
            train_config
        )
        
        self.policy_scheduler = self._create_scheduler(
            self.policy_optimizer,
            train_config
        )
        
        # Loss functions
        self.dropout_loss_fn = DropoutLoss()
        self.trajectory_loss_fn = TrajectoryLoss()
        self.policy_loss_fn = PolicyLoss()
        
        # Loss weights
        self.lambda_dropout = train_config['lambda_dropout']
        self.lambda_trajectory = train_config['lambda_trajectory']
        
        # Training parameters
        self.num_epochs = train_config['num_epochs']
        self.clip_grad_norm = train_config.get('clip_grad_norm', 1.0)
        self.policy_update_freq = train_config.get('policy_update_frequency', 1)
        self.value_update_freq = train_config.get('value_update_frequency', 1)
        
        # Early stopping
        early_stop_config = train_config.get('early_stopping', {})
        self.early_stop_patience = early_stop_config.get('patience', 15)
        self.early_stop_min_delta = early_stop_config.get('min_delta', 1e-4)
        self.early_stop_monitor = early_stop_config.get('monitor', 'val_retention')
        self.early_stop_mode = early_stop_config.get('mode', 'max')
        
        # Tracking
        self.best_val_metric = float('-inf') if self.early_stop_mode == 'max' else float('inf')
        self.epochs_without_improvement = 0
        self.training_history = []
        
        # Logging
        log_config = config.get('logging', {})
        if log_config.get('use_tensorboard', True):
            log_dir = Path(log_config.get('log_dir', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
        
        self.log_frequency = log_config.get('log_frequency', 10)
        
    def _create_optimizer(
        self,
        parameters,
        lr: float,
        weight_decay: float
    ) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_type = self.config['training'].get('optimizer', 'adamw')
        
        if optimizer_type == 'adamw':
            return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(
        self,
        optimizer: optim.Optimizer,
        train_config: Dict
    ):
        """Create learning rate scheduler"""
        scheduler_type = train_config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=train_config.get('scheduler_step_size', 30),
                gamma=train_config.get('scheduler_gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                patience=5,
                factor=0.5
            )
        else:
            return None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'dropout_loss': 0.0,
            'trajectory_loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0
        }
        
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            sequences = batch['sequence'].to(self.device)
            dropout_labels = batch['dropout_label'].to(self.device)
            trajectory_labels = batch['trajectory_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(sequences)
            
            # ===== Predictive Loss (Module I) =====
            
            # Dropout prediction loss
            dropout_loss = self.dropout_loss_fn(
                outputs['dropout_logits'],
                dropout_labels
            )
            
            # Trajectory prediction loss
            trajectory_loss = self.trajectory_loss_fn(
                outputs['trajectory_pred'],
                trajectory_labels.unsqueeze(-1)
            )
            
            # Combined predictive loss
            predictive_loss = (
                self.lambda_dropout * dropout_loss +
                self.lambda_trajectory * trajectory_loss
            )
            
            # Update encoder
            self.encoder_optimizer.zero_grad()
            predictive_loss.backward(retain_graph=True)
            
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_encoder_parameters(),
                    self.clip_grad_norm
                )
            
            self.encoder_optimizer.step()
            
            # ===== Policy Loss (Module II) =====
            
            if batch_idx % self.policy_update_freq == 0:
                # Sample actions
                with torch.no_grad():
                    actions, _, _ = self.model.sample_actions(
                        sequences,
                        budget_ratio=self.config['policy']['budget_ratio'],
                        deterministic=False
                    )
                
                # Recompute outputs for policy gradient
                outputs = self.model(sequences)
                
                # Compute rewards
                rewards = self.model.compute_reward(
                    dropout_labels,
                    trajectory_labels,
                    outputs['trajectory_pred'],
                    actions
                )
                
                # Compute next state values (simplified: use current values)
                next_state_values = outputs['state_values'].detach()
                
                # Compute advantages
                advantages = self.model.compute_advantages(
                    rewards,
                    outputs['state_values'],
                    next_state_values
                )
                
                # Policy gradient loss
                action_probs = outputs['action_probs']
                log_probs = torch.log(action_probs * actions + (1 - action_probs) * (1 - actions) + 1e-8)
                policy_loss = -(log_probs * advantages.detach()).mean()
                
                # Update policy
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_policy_parameters(),
                        self.clip_grad_norm
                    )
                
                self.policy_optimizer.step()
            else:
                policy_loss = torch.tensor(0.0)
            
            # ===== Value Loss =====
            
            if batch_idx % self.value_update_freq == 0:
                # Target values (TD target)
                with torch.no_grad():
                    actions, _, _ = self.model.sample_actions(sequences, deterministic=True)
                    rewards = self.model.compute_reward(
                        dropout_labels,
                        trajectory_labels,
                        outputs['trajectory_pred'],
                        actions
                    )
                    next_state_values = outputs['state_values']
                    target_values = rewards + self.config['policy']['gamma'] * next_state_values
                
                # Value loss (MSE)
                value_loss = F.mse_loss(outputs['state_values'], target_values)
                
                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_value_parameters(),
                        self.clip_grad_norm
                    )
                
                self.value_optimizer.step()
            else:
                value_loss = torch.tensor(0.0)
            
            # Update metrics
            total_loss = predictive_loss + policy_loss + value_loss
            epoch_metrics['loss'] += total_loss.item()
            epoch_metrics['dropout_loss'] += dropout_loss.item()
            epoch_metrics['trajectory_loss'] += trajectory_loss.item()
            epoch_metrics['policy_loss'] += policy_loss.item()
            epoch_metrics['value_loss'] += value_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'dropout': dropout_loss.item(),
                'policy': policy_loss.item()
            })
            
            # Log to tensorboard
            if self.writer and batch_idx % self.log_frequency == 0:
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('train/batch_loss', total_loss.item(), global_step)
                self.writer.add_scalar('train/dropout_loss', dropout_loss.item(), global_step)
                self.writer.add_scalar('train/policy_loss', policy_loss.item(), global_step)
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        all_dropout_probs = []
        all_dropout_labels = []
        all_trajectory_preds = []
        all_trajectory_labels = []
        all_actions = []
        all_rewards = []
        
        val_loss = 0.0
        
        for batch in tqdm(val_loader, desc="Validating"):
            sequences = batch['sequence'].to(self.device)
            dropout_labels = batch['dropout_label'].to(self.device)
            trajectory_labels = batch['trajectory_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(sequences)
            
            # Sample actions
            actions, _, _ = self.model.sample_actions(
                sequences,
                budget_ratio=self.config['policy']['budget_ratio'],
                deterministic=True
            )
            
            # Compute rewards
            rewards = self.model.compute_reward(
                dropout_labels,
                trajectory_labels,
                outputs['trajectory_pred'],
                actions
            )
            
            # Loss
            dropout_loss = self.dropout_loss_fn(outputs['dropout_logits'], dropout_labels)
            trajectory_loss = self.trajectory_loss_fn(
                outputs['trajectory_pred'],
                trajectory_labels.unsqueeze(-1)
            )
            loss = self.lambda_dropout * dropout_loss + self.lambda_trajectory * trajectory_loss
            val_loss += loss.item()
            
            # Collect predictions
            all_dropout_probs.append(outputs['dropout_probs'].cpu().numpy())
            all_dropout_labels.append(dropout_labels.cpu().numpy())
            all_trajectory_preds.append(outputs['trajectory_pred'].cpu().numpy())
            all_trajectory_labels.append(trajectory_labels.cpu().numpy())
            all_actions.append(actions.cpu().numpy())
            all_rewards.append(rewards.cpu().numpy())
        
        # Concatenate all predictions
        dropout_probs = np.concatenate(all_dropout_probs)
        dropout_labels = np.concatenate(all_dropout_labels)
        trajectory_preds = np.concatenate(all_trajectory_preds)
        trajectory_labels = np.concatenate(all_trajectory_labels)
        actions = np.concatenate(all_actions)
        rewards = np.concatenate(all_rewards)
        
        # Compute metrics
        metrics = compute_all_metrics(
            dropout_probs,
            dropout_labels,
            trajectory_preds,
            trajectory_labels,
            actions,
            rewards
        )
        
        metrics['val_loss'] = val_loss / len(val_loader)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print("\n" + "="*70)
        print("STARTING PASTO TRAINING")
        print("="*70)
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update schedulers
            if self.encoder_scheduler is not None:
                if isinstance(self.encoder_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.encoder_scheduler.step(val_metrics.get('auc_pr', 0))
                else:
                    self.encoder_scheduler.step()
            
            if self.policy_scheduler is not None:
                if isinstance(self.policy_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.policy_scheduler.step(val_metrics.get('retention_rate', 0))
                else:
                    self.policy_scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Log epoch results
            print(f"\nEpoch {epoch+1}/{self.num_epochs} - {epoch_time:.2f}s")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Metrics:")
            print(f"  AUC-PR: {val_metrics.get('auc_pr', 0):.4f}")
            print(f"  Recall: {val_metrics.get('recall', 0):.4f}")
            print(f"  F2: {val_metrics.get('f2', 0):.4f}")
            print(f"  Retention: {val_metrics.get('retention_rate', 0):.4f}")
            print(f"  Mean Reward: {val_metrics.get('mean_reward', 0):.4f}")
            
            # Tensorboard logging
            if self.writer:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f'train/{key}', value, epoch)
                for key, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'val/{key}', value, epoch)
            
            # Check for improvement
            current_metric = val_metrics.get(self.early_stop_monitor, 0)
            
            if self.early_stop_mode == 'max':
                improved = current_metric > self.best_val_metric + self.early_stop_min_delta
            else:
                improved = current_metric < self.best_val_metric - self.early_stop_min_delta
            
            if improved:
                self.best_val_metric = current_metric
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"✓ New best model saved! ({self.early_stop_monitor}: {current_metric:.4f})")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epochs")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['training'].get('save_frequency', 5) == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Store history
            self.training_history.append({
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict,
        is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if is_best:
            path = self.save_dir / 'pasto_best.pth'
        else:
            path = self.save_dir / f'pasto_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']


if __name__ == "__main__":
    print("Trainer module loaded successfully")
