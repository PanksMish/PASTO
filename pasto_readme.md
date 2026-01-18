# PASTO: Policy-Aware Sequential Trajectory Optimization for Equitable Student Dropout Intervention

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of "PASTO: Policy-Aware Sequential Optimization for Equitable Student Dropout Intervention" (IJCAI 2024).

## Overview

PASTO is a framework that integrates deep temporal trajectory modeling with restless multi-armed bandit (RMAB)-based policy optimization for student dropout intervention. The system learns representations from high-dimensional trajectories and performs sequential resource allocation under strict budget constraints while maintaining equity considerations.

## Key Features

- **Deep Trajectory Modeling**: LSTM/Transformer-based encoders for temporal student data
- **RMAB-based Policy Optimization**: Learned Whittle indices for sequential intervention allocation
- **Incentive-Aligned Prioritization**: Monotone scoring functions for strategic robustness
- **Equity-Aware Allocation**: Balanced performance across underprivileged student groups
- **Budget-Constrained Planning**: Efficient resource allocation under real-world constraints

## Architecture

```
PASTO
├── Module I: Predictive State Modeling
│   ├── Trajectory Encoder (LSTM/Transformer)
│   ├── Dropout Risk Predictor
│   └── Long-term Outcome Predictor
│
└── Module II: Prescriptive Policy Optimization
    ├── Parameterized Whittle Index Learning
    ├── Advantage Estimation (Temporal Difference)
    └── Budget-Constrained Greedy Allocation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pasto.git
cd pasto

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download Datasets

```bash
# Download OULAD dataset
python scripts/download_data.py --dataset oulad

# Download Indian school dataset
python scripts/download_data.py --dataset indian_school
```

### 2. Preprocess Data

```bash
python src/data/preprocess.py --dataset oulad --output data/processed/
```

### 3. Train PASTO

```bash
python train.py --config configs/pasto_oulad.yaml --gpu 0
```

### 4. Evaluate

```bash
python evaluate.py --checkpoint checkpoints/pasto_best.pth --dataset oulad
```

## Project Structure

```
pasto/
├── configs/                    # Configuration files
│   ├── pasto_oulad.yaml
│   ├── pasto_indian.yaml
│   └── baseline_configs/
├── data/                       # Data directory
│   ├── raw/
│   ├── processed/
│   └── datasets/
├── src/                        # Source code
│   ├── models/                 # Model implementations
│   │   ├── trajectory_encoder.py
│   │   ├── dropout_predictor.py
│   │   ├── whittle_index.py
│   │   └── pasto.py
│   ├── data/                   # Data processing
│   │   ├── dataset_loader.py
│   │   ├── preprocess.py
│   │   └── augmentation.py
│   ├── training/               # Training logic
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── optimizers.py
│   ├── evaluation/             # Evaluation metrics
│   │   ├── metrics.py
│   │   ├── policy_evaluator.py
│   │   └── equity_metrics.py
│   └── utils/                  # Utilities
│       ├── logger.py
│       ├── visualization.py
│       └── helpers.py
├── baselines/                  # Baseline implementations
│   ├── xgboost_baseline.py
│   ├── random_forest_baseline.py
│   ├── dqn_baseline.py
│   └── whittle_rmab_baseline.py
├── scripts/                    # Utility scripts
│   ├── download_data.py
│   └── run_experiments.py
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks
│   ├── exploratory_analysis.ipynb
│   └── result_visualization.ipynb
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Datasets

### Supported Datasets

1. **OULAD (Open University Learning Analytics Dataset)**
   - 32,593 students
   - ~10.6M interaction events
   - Weekly temporal resolution
   - [Download Link](https://analyse.kmi.open.ac.uk/open_dataset)

2. **Indian School Dataset**
   - Annual academic records
   - Multi-year trajectories
   - Socio-economic features

3. **Custom Datasets**
   - See `docs/custom_dataset_guide.md` for format specifications

## Configuration

Edit `configs/pasto_oulad.yaml` to customize training:

```yaml
model:
  encoder_type: "lstm"  # or "transformer"
  hidden_dim: 128
  num_layers: 2
  dropout: 0.3

training:
  batch_size: 256
  learning_rate: 1e-4
  num_epochs: 100
  
policy:
  budget_ratio: 0.1
  alpha: 0.7  # retention vs trajectory trade-off
  
equity:
  protected_groups: ["low_ses"]
  fairness_weight: 0.3
```

## Reproducing Paper Results

```bash
# Run full experimental suite
bash scripts/run_all_experiments.sh

# Run specific experiment
python train.py --config configs/experiments/table1_comparison.yaml

# Generate figures
python scripts/generate_figures.py --results results/
```

## Evaluation Metrics

- **Predictive Metrics**: AUC-PR, Recall, F2-Score, KS-Statistic
- **Policy Metrics**: Cumulative Regret, Retention Rate, Trajectory Gain
- **Cost Metrics**: Intervention Efficiency Ratio (IER)
- **Equity Metrics**: Equity-Adjusted Retention Gain (EARG)

## Results Summary

| Model | AUC-PR | Recall | Regret ↓ | IER ↑ | EARG ↑ |
|-------|--------|--------|----------|-------|--------|
| PASTO | 0.89 | 0.88 | 0.42 | 0.61 | 0.91 |
| RL-DQN | 0.85 | 0.78 | 0.67 | 0.47 | 0.58 |
| Whittle-RMAB | 0.83 | 0.76 | 0.71 | 0.45 | 0.42 |
| XGBoost | 0.82 | 0.70 | - | - | 0.38 |

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{pasto2024,
  title={PASTO: Policy-Aware Sequential Trajectory Optimization for Equitable Student Dropout Intervention},
  author={Anonymous},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Email: [your-email@institution.edu]

## Acknowledgments

- Open University Learning Analytics Dataset (OULAD) providers
- Research supported by [Funding Agency]
- Built with PyTorch, scikit-learn, and pandas

## Changelog

### Version 1.0.0 (2024-01)
- Initial release
- OULAD dataset support
- Complete PASTO implementation
- Baseline comparisons

---

**Note**: This is a research implementation. For production deployment, additional validation and testing are recommended.
