"""
Training script for Whittle RMAB baseline
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from whittle_rmab_baseline import WhittleRMABBaseline
from src.data.dataset_loader import OULADDatasetLoader
from src.data.preprocess import preprocess_pipeline, FeatureEngineer
from src.evaluation.metrics import compute_all_metrics, print_metrics_table
from src.utils.helpers import load_config, set_seed
from sklearn.model_selection import train_test_split


def main(args):
    """Main training function for Whittle RMAB"""
    
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
                'num_states': 24,
                'gamma': 0.95
            },
            'experiment': {'seed': 42}
        }
    
    set_seed(config['experiment']['seed'])
    
    print("\n" + "="*70)
    print("TRAINING WHITTLE RMAB BASELINE")
    print("="*70)
    
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
    
    # Split data
    train_idx, test_idx = train_test_split(
        np.arange(len(sequences)), test_size=0.3,
        stratify=dropout_labels, random_state=config['experiment']['seed']
    )
    
    # Create synthetic actions and rewards for training
    actions = np.random.randint(0, 2, len(sequences))
    rewards = (1 - dropout_labels).astype(float)
    
    # Initialize and fit RMAB
    print("\nInitializing Whittle RMAB...")
    model = WhittleRMABBaseline(
        num_states=config['model']['num_states'],
        config=config['model']
    )
    
    model.fit(
        sequences[train_idx],
        actions[train_idx],
        rewards[train_idx]
    )
    
    # Evaluate
    print("\nEvaluating...")
    test_actions = model.predict(sequences[test_idx], budget_ratio=0.1)
    
    # Discretize states for dropout prediction
    N, T, F = sequences[test_idx].shape
    features_flat = sequences[test_idx].reshape(N, T * F)
    states = model.discretize_states(features_flat)
    
    # Use Whittle indices as dropout risk proxy
    dropout_probs = model.whittle_indices[states]
    dropout_probs = (dropout_probs - dropout_probs.min()) / (dropout_probs.max() - dropout_probs.min() + 1e-8)
    
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
    output_dir = Path('results/whittle_rmab')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save Whittle indices
    np.save(output_dir / 'whittle_indices.npy', model.whittle_indices)
    
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Whittle RMAB baseline')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=-1)
    
    args = parser.parse_args()
    main(args)
