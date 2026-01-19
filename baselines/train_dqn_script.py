"""
Training script for DQN baseline
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from dqn_baseline import DQNBaseline
from src.data.dataset_loader import OULADDatasetLoader
from src.data.preprocess import preprocess_pipeline, FeatureEngineer
from src.evaluation.metrics import compute_all_metrics, print_metrics_table
from src.utils.helpers import load_config, set_seed
from sklearn.model_selection import train_test_split


def main(args):
    """Main training function for DQN"""
    
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            'data': {
                'data_dir': 'data/raw/oulad',
                'sequence_length': 30,
                'normalize': True
            },
            'model': {
                'gamma': 0.95,
                'epsilon_start': 1.0,
                'learning_rate': 1e-3,
                'batch_size': 64
            },
            'training': {
                'num_epochs': 100
            },
            'experiment': {'seed': 42}
        }
    
    set_seed(config['experiment']['seed'])
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*70)
    print("TRAINING DQN BASELINE")
    print("="*70)
    print(f"Device: {device}")
    
    # Load data
    loader = OULADDatasetLoader(
        data_dir=config['data']['data_dir'],
        sequence_length=config['data']['sequence_length']
    )
    
    temporal_df = loader.create_temporal_features()
    engineer = FeatureEngineer(config=config['data'])
    temporal_df = engineer.engineer_features(temporal_df)
    
    feature_columns = ['sum_click', 'days_active', 'avg_score', 'num_assessments']
    
    sequences, dropout_labels, trajectory_labels, student_ids = \
        loader.create_sequences(temporal_df, feature_columns)
    
    sequences, dropout_labels, trajectory_labels = preprocess_pipeline(
        sequences, dropout_labels, trajectory_labels,
        config=config['data'], is_training=True
    )
    
    # Flatten sequences to states
    N, T, F = sequences.shape
    states = sequences.reshape(N, T * F)
    
    # Split data
    train_idx, test_idx = train_test_split(
        np.arange(len(states)), test_size=0.3,
        stratify=dropout_labels, random_state=config['experiment']['seed']
    )
    
    # Create synthetic actions and rewards for offline training
    actions = np.random.randint(0, 2, len(states))
    rewards = (1 - dropout_labels).astype(float)  # Reward for retention
    next_states = states.copy()  # Simplified
    dones = dropout_labels.copy()
    
    # Initialize DQN
    state_dim = T * F
    dqn = DQNBaseline(state_dim=state_dim, config=config['model'], device=device)
    
    # Train offline
    print("\nTraining DQN offline...")
    losses = dqn.train_offline(
        states[train_idx],
        actions[train_idx],
        rewards[train_idx],
        next_states[train_idx],
        dones[train_idx],
        num_epochs=config['training']['num_epochs']
    )
    
    # Evaluate
    print("\nEvaluating...")
    test_actions = dqn.select_actions_batch(states[test_idx], budget_ratio=0.1)
    
    # Compute dropout predictions (use state values as proxy)
    with torch.no_grad():
        states_tensor = torch.FloatTensor(states[test_idx]).to(device)
        q_values = dqn.q_network(states_tensor)
        dropout_probs = torch.sigmoid(-q_values[:, 1]).cpu().numpy()
    
    metrics = compute_all_metrics(
        dropout_probs,
        dropout_labels[test_idx],
        trajectory_labels[test_idx],
        trajectory_labels[test_idx],
        test_actions,
        rewards[test_idx]
    )
    
    print_metrics_table(metrics)
    
    # Save
    output_dir = Path('results/dqn')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    dqn.save(output_dir / 'dqn_model.pth')
    
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN baseline')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    main(args)
