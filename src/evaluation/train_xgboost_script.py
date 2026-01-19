"""
Training script for XGBoost baseline
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from xgboost_baseline import XGBoostBaseline, train_xgboost_baseline
from src.data.dataset_loader import OULADDatasetLoader, create_dataloaders
from src.data.preprocess import preprocess_pipeline, FeatureEngineer
from src.evaluation.metrics import compute_all_metrics, print_metrics_table
from src.utils.helpers import load_config, set_seed


def main(args):
    """Main training function for XGBoost"""
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            'data': {
                'dataset_name': 'oulad',
                'data_dir': 'data/raw/oulad',
                'sequence_length': 30,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'use_smote': True,
                'sampling_strategy': 0.5,
                'normalize': True
            },
            'model': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100
            },
            'experiment': {'seed': 42}
        }
    
    # Set seed
    set_seed(config['experiment']['seed'])
    
    print("\n" + "="*70)
    print("TRAINING XGBOOST BASELINE")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    loader = OULADDatasetLoader(
        data_dir=config['data']['data_dir'],
        sequence_length=config['data']['sequence_length']
    )
    
    temporal_df = loader.create_temporal_features()
    
    # Engineer features
    engineer = FeatureEngineer(config=config['data'])
    temporal_df = engineer.engineer_features(temporal_df)
    
    feature_columns = [
        'sum_click', 'days_active', 'avg_score',
        'num_assessments', 'studied_credits'
    ]
    
    sequences, dropout_labels, trajectory_labels, student_ids = \
        loader.create_sequences(temporal_df, feature_columns)
    
    # Preprocess
    sequences, dropout_labels, trajectory_labels = preprocess_pipeline(
        sequences, dropout_labels, trajectory_labels,
        config=config['data'], is_training=True
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_idx, temp_idx = train_test_split(
        np.arange(len(sequences)),
        test_size=0.3,
        stratify=dropout_labels,
        random_state=config['experiment']['seed']
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=dropout_labels[temp_idx],
        random_state=config['experiment']['seed']
    )
    
    X_train, y_train = sequences[train_idx], dropout_labels[train_idx]
    X_val, y_val = sequences[val_idx], dropout_labels[val_idx]
    X_test, y_test = sequences[test_idx], dropout_labels[test_idx]
    
    # Train model
    print("\nTraining XGBoost...")
    model = train_xgboost_baseline(X_train, y_train, X_val, y_val, config['model'])
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # Compute metrics
    metrics = compute_all_metrics(
        y_pred_proba, y_test,
        trajectory_labels[test_idx], trajectory_labels[test_idx],
        y_pred, np.zeros_like(y_pred)  # No policy actions for baseline
    )
    
    print_metrics_table(metrics)
    
    # Save results
    output_dir = Path('results/xgboost')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    model.save(output_dir / 'xgboost_model.pkl')
    
    print(f"\n✓ Results saved to {output_dir}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost baseline')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU (not used)')
    
    args = parser.parse_args()
    main(args)
