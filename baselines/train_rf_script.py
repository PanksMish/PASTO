"""
Training script for Random Forest baseline
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from random_forest_baseline import RandomForestBaseline
from src.data.dataset_loader import OULADDatasetLoader
from src.data.preprocess import preprocess_pipeline, FeatureEngineer
from src.evaluation.metrics import compute_all_metrics, print_metrics_table
from src.utils.helpers import load_config, set_seed
from sklearn.model_selection import train_test_split


def main(args):
    """Main training function for Random Forest"""
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            'data': {
                'dataset_name': 'oulad',
                'data_dir': 'data/raw/oulad',
                'sequence_length': 30,
                'normalize': True,
                'use_smote': True
            },
            'model': {
                'n_estimators': 100,
                'max_depth': None,
                'class_weight': 'balanced'
            },
            'experiment': {'seed': 42}
        }
    
    set_seed(config['experiment']['seed'])
    
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST BASELINE")
    print("="*70)
    
    # Load and preprocess data (similar to XGBoost)
    loader = OULADDatasetLoader(
        data_dir=config['data']['data_dir'],
        sequence_length=config['data']['sequence_length']
    )
    
    temporal_df = loader.create_temporal_features()
    engineer = FeatureEngineer(config=config['data'])
    temporal_df = engineer.engineer_features(temporal_df)
    
    feature_columns = [
        'sum_click', 'days_active', 'avg_score',
        'num_assessments', 'studied_credits'
    ]
    
    sequences, dropout_labels, trajectory_labels, student_ids = \
        loader.create_sequences(temporal_df, feature_columns)
    
    sequences, dropout_labels, trajectory_labels = preprocess_pipeline(
        sequences, dropout_labels, trajectory_labels,
        config=config['data'], is_training=True
    )
    
    # Split data
    train_idx, temp_idx = train_test_split(
        np.arange(len(sequences)), test_size=0.3,
        stratify=dropout_labels, random_state=config['experiment']['seed']
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        stratify=dropout_labels[temp_idx],
        random_state=config['experiment']['seed']
    )
    
    X_train, y_train = sequences[train_idx], dropout_labels[train_idx]
    X_test, y_test = sequences[test_idx], dropout_labels[test_idx]
    
    # Train
    print("\nTraining Random Forest...")
    model = RandomForestBaseline(config['model'])
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating...")
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    metrics = compute_all_metrics(
        y_pred_proba, y_test,
        trajectory_labels[test_idx], trajectory_labels[test_idx],
        y_pred, np.zeros_like(y_pred)
    )
    
    print_metrics_table(metrics)
    
    # Save
    output_dir = Path('results/random_forest')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    model.save(output_dir / 'random_forest_model.pkl')
    
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Random Forest baseline')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=-1)
    
    args = parser.parse_args()
    main(args)
