"""
Unit tests for PASTO models
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.trajectory_encoder import LSTMEncoder, TransformerEncoder, StateDiscretizer
from src.models.dropout_predictor import DropoutRiskPredictor, TrajectoryOutcomePredictor
from src.models.whittle_index import WhittleIndexNetwork, ValueNetwork, PolicyNetwork
from src.models.pasto import PASTO


@pytest.fixture
def config():
    """Test configuration"""
    return {
        'model': {
            'encoder': {
                'type': 'lstm',
                'input_dim': 64,
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.3,
                'bidirectional': True
            },
            'transformer': {
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dim_feedforward': 512,
                'dropout': 0.3
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
            'budget_ratio': 0.1
        },
        'training': {
            'critic': {
                'hidden_dims': [128, 64],
                'dropout': 0.1
            }
        }
    }


class TestTrajectoryEncoder:
    """Test trajectory encoder modules"""
    
    def test_lstm_encoder(self):
        """Test LSTM encoder"""
        batch_size, seq_len, input_dim = 32, 30, 64
        
        encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        seq_out, final_hidden = encoder(x)
        
        assert seq_out.shape == (batch_size, seq_len, 256)  # 128 * 2 (bidirectional)
        assert final_hidden.shape == (batch_size, 256)
    
    def test_transformer_encoder(self):
        """Test Transformer encoder"""
        batch_size, seq_len, input_dim = 32, 30, 64
        
        encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_encoder_layers=3
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        seq_out, pooled = encoder(x)
        
        assert seq_out.shape == (batch_size, seq_len, 128)
        assert pooled.shape == (batch_size, 128)
    
    def test_state_discretizer(self):
        """Test state discretizer"""
        batch_size, hidden_dim = 32, 256
        
        discretizer = StateDiscretizer(
            input_dim=hidden_dim,
            num_risk_bins=4,
            num_engagement_bins=4
        )
        
        h = torch.randn(batch_size, hidden_dim)
        states = discretizer(h)
        
        assert states.shape == (batch_size,)
        assert states.min() >= 0
        assert states.max() < 24


class TestDropoutPredictor:
    """Test dropout prediction modules"""
    
    def test_dropout_predictor(self):
        """Test dropout risk predictor"""
        batch_size, input_dim = 32, 256
        
        predictor = DropoutRiskPredictor(
            input_dim=input_dim,
            hidden_dims=[256, 128, 64]
        )
        
        h = torch.randn(batch_size, input_dim)
        logits = predictor(h)
        probs = predictor.predict_proba(h)
        
        assert logits.shape == (batch_size, 1)
        assert probs.shape == (batch_size,)
        assert (probs >= 0).all() and (probs <= 1).all()
    
    def test_trajectory_predictor(self):
        """Test trajectory outcome predictor"""
        batch_size, input_dim = 32, 256
        
        predictor = TrajectoryOutcomePredictor(
            input_dim=input_dim,
            hidden_dims=[256, 128]
        )
        
        h = torch.randn(batch_size, input_dim)
        preds = predictor(h)
        
        assert preds.shape == (batch_size, 1)


class TestWhittleIndex:
    """Test Whittle index modules"""
    
    def test_whittle_index_network(self):
        """Test Whittle index network"""
        batch_size, state_dim = 32, 256
        
        network = WhittleIndexNetwork(
            state_dim=state_dim,
            hidden_dims=[128, 64, 32]
        )
        
        states = torch.randn(batch_size, state_dim)
        indices = network(states)
        
        assert indices.shape == (batch_size, 1)
    
    def test_value_network(self):
        """Test value network"""
        batch_size, state_dim = 32, 256
        
        network = ValueNetwork(
            state_dim=state_dim,
            hidden_dims=[128, 64]
        )
        
        states = torch.randn(batch_size, state_dim)
        values = network(states)
        
        assert values.shape == (batch_size, 1)
    
    def test_policy_network(self):
        """Test policy network"""
        batch_size, state_dim = 32, 256
        
        whittle_net = WhittleIndexNetwork(state_dim=state_dim)
        policy = PolicyNetwork(whittle_net)
        
        states = torch.randn(batch_size, state_dim)
        probs = policy(states)
        actions, log_probs = policy.sample_actions(states)
        
        assert probs.shape == (batch_size,)
        assert actions.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)


class TestPASTO:
    """Test complete PASTO model"""
    
    def test_pasto_forward(self, config):
        """Test PASTO forward pass"""
        batch_size, seq_len, input_dim = 32, 30, 64
        
        model = PASTO(config)
        sequences = torch.randn(batch_size, seq_len, input_dim)
        
        outputs = model(sequences)
        
        assert 'dropout_probs' in outputs
        assert 'trajectory_pred' in outputs
        assert 'whittle_indices' in outputs
        assert 'priority_scores' in outputs
        
        assert outputs['dropout_probs'].shape == (batch_size,)
        assert outputs['trajectory_pred'].shape == (batch_size, 1)
        assert outputs['whittle_indices'].shape == (batch_size,)
    
    def test_pasto_action_sampling(self, config):
        """Test action sampling"""
        batch_size, seq_len, input_dim = 32, 30, 64
        
        model = PASTO(config)
        sequences = torch.randn(batch_size, seq_len, input_dim)
        
        actions, priorities, outputs = model.sample_actions(
            sequences,
            budget_ratio=0.1,
            deterministic=True
        )
        
        assert actions.shape == (batch_size,)
        assert priorities.shape == (batch_size,)
        
        # Check budget constraint
        num_interventions = actions.sum().item()
        expected_interventions = int(batch_size * 0.1)
        assert num_interventions <= expected_interventions + 1
    
    def test_pasto_reward_computation(self, config):
        """Test reward computation"""
        batch_size = 32
        
        model = PASTO(config)
        
        dropout_labels = torch.randint(0, 2, (batch_size,))
        trajectory_labels = torch.randn(batch_size)
        trajectory_pred = torch.randn(batch_size, 1)
        actions = torch.randint(0, 2, (batch_size,))
        
        rewards = model.compute_reward(
            dropout_labels,
            trajectory_labels,
            trajectory_pred,
            actions
        )
        
        assert rewards.shape == (batch_size,)


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end(self, config):
        """Test end-to-end pipeline"""
        batch_size, seq_len, input_dim = 16, 30, 64
        
        # Create model
        model = PASTO(config)
        model.eval()
        
        # Generate data
        sequences = torch.randn(batch_size, seq_len, input_dim)
        dropout_labels = torch.randint(0, 2, (batch_size,))
        trajectory_labels = torch.randn(batch_size)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(sequences)
            
            # Sample actions
            actions, _, _ = model.sample_actions(sequences, budget_ratio=0.1)
            
            # Compute rewards
            rewards = model.compute_reward(
                dropout_labels,
                trajectory_labels,
                outputs['trajectory_pred'],
                actions
            )
        
        # Check shapes
        assert outputs['dropout_probs'].shape == (batch_size,)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        
        # Check values are reasonable
        assert (outputs['dropout_probs'] >= 0).all()
        assert (outputs['dropout_probs'] <= 1).all()
        assert (actions >= 0).all() and (actions <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
