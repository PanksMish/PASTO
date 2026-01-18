"""
Main Training Script for PASTO
Usage: python train.py --config configs/pasto_oulad.yaml --gpu 0
"""

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.pasto import PASTO
from src.data.dataset_loader import (
    OULADDatasetLoader,
    create_dataloaders
)
from src.data.preprocess import preprocess_pipeline, FeatureEngineer
from src.training.trainer import PASTOTrainer


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict):
    """
    Prepare dataset and dataloaders
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, feature_dim)
    """
    print("\n" + "="*70)
    print("DATA PREPARATION")
    print("="*70)
    
    data_config = config['data']
    dataset_name = data_config['dataset_name']
    
    # Load appropriate dataset
    if dataset_name == 'oulad':
        print("\nLoading OULAD dataset...")
        loader = OULADDatasetLoader(
            data_dir=data_config['data_dir'],
            sequence_length=data_config['sequence_length']
        )
        
        # Create temporal features
        temporal_df = loader.create_temporal_features()
        
        # Engineer features
        print("\nEngineering features...")
        engineer = FeatureEngineer(config=data_config)
        temporal_df = engineer.engineer_features(temporal_df)
        
        # Select features
        feature_columns = data_config.get('features', [
            'sum_click', 'days_active', 'avg_score',
            'num_assessments', 'studied_credits'
        ])
        
        # Add engineered features
        available_features = [col for col in temporal_df.columns 
                            if any(feat in col for feat in feature_columns)]
        
        print(f"\nUsing {len(available_features)} features")
        
        # Create sequences
        sequences, dropout_labels, trajectory_labels, student_ids = \
            loader.create_sequences(temporal_df, available_features)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Preprocess data
    sequences, dropout_labels, trajectory_labels = preprocess_pipeline(
        sequences,
        dropout_labels,
        trajectory_labels,
        config=data_config,
        is_training=True
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        sequences,
        dropout_labels,
        trajectory_labels,
        student_ids,
        batch_size=config['training']['batch_size'],
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio'],
        num_workers=config['experiment'].get('num_workers', 4),
        seed=config['experiment']['seed']
    )
    
    feature_dim = sequences.shape[-1]
    
    print(f"\nFeature dimension: {feature_dim}")
    print("="*70)
    
    return train_loader, val_loader, test_loader, feature_dim


def main(args):
    """Main training function"""
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.gpu is not None:
        config['experiment']['device'] = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    
    if args.seed is not None:
        config['experiment']['seed'] = args.seed
    
    # Set device
    device = config['experiment']['device']
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
        config['experiment']['device'] = 'cpu'
    
    print("\n" + "="*70)
    print("PASTO TRAINING")
    print("="*70)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Device: {device}")
    print(f"Seed: {config['experiment']['seed']}")
    
    # Set random seed
    set_seed(config['experiment']['seed'])
    
    # Prepare data
    train_loader, val_loader, test_loader, feature_dim = prepare_data(config)
    
    # Update config with actual feature dimension
    config['model']['encoder']['input_dim'] = feature_dim
    
    # Initialize model
    print("\n" + "="*70)
    print("MODEL INITIALIZATION")
    print("="*70)
    
    model = PASTO(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    save_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
    save_dir = save_dir / config['experiment']['name']
    save_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = PASTOTrainer(
        model=model,
        config=config,
        device=device,
        save_dir=str(save_dir)
    )
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    test_metrics = trainer.validate(test_loader)
    
    from src.evaluation.metrics import print_metrics_table
    print_metrics_table(test_metrics)
    
    # Save final metrics
    import json
    metrics_path = save_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\n✓ Test metrics saved to: {metrics_path}")
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PASTO model')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pasto_oulad.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device ID (-1 for CPU, None to use config default)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (None to use config default)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    
    args = parser.parse_args()
    
    main(args)
