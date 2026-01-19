# PASTO Project Structure

Complete file structure for the PASTO implementation.

```
pasto/
│
├── README.md                           # Main project documentation
├── LICENSE                             # MIT License
├── CONTRIBUTING.md                     # Contribution guidelines
├── PROJECT_STRUCTURE.md               # This file
├── .gitignore                         # Git ignore rules
├── requirements.txt                    # Python dependencies
├── setup.py                           # Package installation script
│
├── configs/                           # Configuration files
│   ├── pasto_oulad.yaml              # Main PASTO config for OULAD
│   ├── pasto_indian.yaml             # Config for Indian school dataset
│   └── baseline_configs/             # Baseline configurations
│       ├── xgboost_config.yaml
│       ├── random_forest_config.yaml
│       ├── dqn_config.yaml
│       └── whittle_rmab_config.yaml
│
├── data/                              # Data directory
│   ├── raw/                          # Raw datasets
│   │   ├── oulad/                    # OULAD dataset
│   │   ├── indian_school/            # Indian school dataset
│   │   └── synthetic/                # Synthetic data
│   ├── processed/                    # Processed datasets
│   └── datasets/                     # Dataset metadata
│
├── src/                               # Source code
│   ├── __init__.py
│   │
│   ├── models/                        # Model implementations
│   │   ├── __init__.py
│   │   ├── trajectory_encoder.py     # LSTM/Transformer encoders
│   │   ├── dropout_predictor.py      # Dropout prediction models
│   │   ├── whittle_index.py          # RMAB policy components
│   │   └── pasto.py                  # Main PASTO model
│   │
│   ├── data/                          # Data processing
│   │   ├── __init__.py
│   │   ├── dataset_loader.py         # Dataset loading utilities
│   │   ├── preprocess.py             # Preprocessing pipeline
│   │   └── augmentation.py           # Data augmentation
│   │
│   ├── training/                      # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py                # Main training loop
│   │   ├── losses.py                 # Loss functions
│   │   └── optimizers.py             # Custom optimizers
│   │
│   ├── evaluation/                    # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── metrics.py                # All evaluation metrics
│   │   ├── policy_evaluator.py       # Policy-specific evaluation
│   │   └── equity_metrics.py         # Fairness metrics
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── logger.py                 # Logging utilities
│       ├── visualization.py          # Visualization tools
│       └── helpers.py                # Helper functions
│
├── baselines/                         # Baseline implementations
│   ├── __init__.py
│   ├── xgboost_baseline.py           # XGBoost baseline
│   ├── random_forest_baseline.py     # Random Forest baseline
│   ├── dqn_baseline.py               # DQN baseline
│   ├── whittle_rmab_baseline.py      # Classical Whittle RMAB
│   ├── train_xgboost.py              # XGBoost training script
│   ├── train_random_forest.py        # RF training script
│   ├── train_dqn.py                  # DQN training script
│   └── train_whittle_rmab.py         # RMAB training script
│
├── scripts/                           # Utility scripts
│   ├── download_data.py              # Data download/generation
│   ├── run_experiments.py            # Run all experiments
│   └── generate_figures.py           # Generate paper figures
│
├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_models.py                # Model tests
│   ├── test_data.py                  # Data processing tests
│   ├── test_training.py              # Training tests
│   └── test_metrics.py               # Metrics tests
│
├── notebooks/                         # Jupyter notebooks
│   ├── quickstart_tutorial.py        # Quick start guide
│   ├── exploratory_analysis.ipynb    # Data exploration
│   ├── result_visualization.ipynb    # Results visualization
│   └── ablation_studies.ipynb        # Ablation experiments
│
├── checkpoints/                       # Model checkpoints
│   ├── pasto_best.pth
│   └── pasto_epoch_*.pth
│
├── logs/                              # Training logs
│   ├── tensorboard/
│   └── training_logs/
│
├── results/                           # Experiment results
│   ├── pasto/
│   ├── xgboost/
│   ├── random_forest/
│   ├── dqn/
│   ├── whittle_rmab/
│   └── comparison.csv
│
├── figures/                           # Generated figures
│   ├── precision_recall_curve.png
│   ├── roc_curve.png
│   ├── training_curves.png
│   ├── retention_over_time.png
│   └── model_comparison.png
│
├── docs/                              # Additional documentation
│   ├── custom_dataset_guide.md
│   ├── hyperparameter_tuning.md
│   └── deployment_guide.md
│
├── train.py                           # Main training script
├── evaluate.py                        # Evaluation script
└── requirements-dev.txt               # Development dependencies
```

## File Descriptions

### Core Files

1. **README.md** - Complete project documentation with installation, usage, and examples
2. **requirements.txt** - All Python package dependencies
3. **setup.py** - Package installation and distribution setup
4. **train.py** - Main script to train PASTO model
5. **evaluate.py** - Script to evaluate trained models

### Configuration Files

- **configs/pasto_oulad.yaml** - Complete configuration for PASTO on OULAD dataset
- **configs/baseline_configs/** - Configurations for all baseline models

### Source Code (src/)

#### Models (src/models/)
- **trajectory_encoder.py** - LSTM and Transformer encoders for temporal sequences
- **dropout_predictor.py** - Dropout risk and trajectory outcome predictors
- **whittle_index.py** - Whittle index networks, value networks, policy networks
- **pasto.py** - Complete PASTO model integrating all components

#### Data (src/data/)
- **dataset_loader.py** - Dataset loading for OULAD, Indian school, custom datasets
- **preprocess.py** - Feature engineering, normalization, SMOTE, augmentation

#### Training (src/training/)
- **trainer.py** - Complete training loop with policy gradient optimization
- **losses.py** - All loss functions (dropout, trajectory, policy, composite)

#### Evaluation (src/evaluation/)
- **metrics.py** - All metrics from paper (AUC-PR, F2, IER, EARG, etc.)

#### Utils (src/utils/)
- **logger.py** - Logging utilities
- **visualization.py** - All plotting functions
- **helpers.py** - Helper functions (timers, config loading, etc.)

### Baselines (baselines/)

- **xgboost_baseline.py** - XGBoost implementation
- **random_forest_baseline.py** - Random Forest implementation
- **dqn_baseline.py** - Deep Q-Network implementation
- **whittle_rmab_baseline.py** - Classical Whittle RMAB implementation

### Scripts (scripts/)

- **download_data.py** - Download/generate datasets
- **run_experiments.py** - Run all experiments from paper
- **generate_figures.py** - Generate all paper figures

### Tests (tests/)

- **test_models.py** - Unit tests for all models
- **test_data.py** - Tests for data processing
- **test_training.py** - Tests for training pipeline
- **test_metrics.py** - Tests for evaluation metrics

### Notebooks (notebooks/)

- **quickstart_tutorial.py** - Complete tutorial from data loading to evaluation
- **exploratory_analysis.ipynb** - Data exploration and visualization
- **result_visualization.ipynb** - Results analysis and plotting

## Usage Examples

### 1. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python scripts/download_data.py --dataset synthetic

# Train PASTO
python train.py --config configs/pasto_oulad.yaml --gpu 0

# Evaluate
python evaluate.py --checkpoint checkpoints/pasto_best.pth --visualize
```

### 2. Run All Experiments

```bash
python scripts/run_experiments.py --all --gpu 0
```

### 3. Run Individual Baseline

```bash
python baselines/train_xgboost.py --config configs/baseline_configs/xgboost_config.yaml
```

### 4. Run Tests

```bash
pytest tests/ -v --cov=src
```

## Key Features

✅ **Complete Implementation** - All paper components implemented
✅ **Modular Design** - Easy to extend and modify
✅ **Comprehensive Testing** - Unit tests for all modules
✅ **Multiple Baselines** - XGBoost, RF, DQN, Whittle RMAB
✅ **Rich Evaluation** - All metrics from paper
✅ **Visualization** - Extensive plotting utilities
✅ **Documentation** - Detailed docstrings and tutorials
✅ **Reproducible** - Seed setting and deterministic training

## Next Steps

1. Download real OULAD dataset or use synthetic data
2. Follow quickstart tutorial
3. Train PASTO and baselines
4. Compare results
5. Customize for your use case

For detailed instructions, see README.md
