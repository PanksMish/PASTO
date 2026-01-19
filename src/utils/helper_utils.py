"""
Helper utilities for PASTO
"""

import torch
import numpy as np
import random
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from datetime import datetime


def set_seed(seed: int):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML config
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, save_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save YAML
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_json(path: str) -> Dict:
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, path: str):
    """Save to JSON file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get torch device
    
    Args:
        gpu_id: GPU ID (None for CPU, -1 for auto)
        
    Returns:
        Torch device
    """
    if gpu_id is None or gpu_id < 0:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        if torch.cuda.is_available():
            return torch.device(f'cuda:{gpu_id}')
        else:
            print(f"GPU {gpu_id} not available, using CPU")
            return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def format_time(seconds: float) -> str:
    """
    Format seconds to readable time string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


class Timer:
    """Simple timer context manager"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"{self.name} took {format_time(elapsed)}")


def create_experiment_dir(base_dir: str, exp_name: Optional[str] = None) -> Path:
    """
    Create experiment directory with timestamp
    
    Args:
        base_dir: Base directory
        exp_name: Experiment name
        
    Returns:
        Path to experiment directory
    """
    if exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"exp_{timestamp}"
    
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)
    (exp_dir / 'figures').mkdir(exist_ok=True)
    
    return exp_dir


def get_git_info() -> Dict[str, str]:
    """
    Get git repository information
    
    Returns:
        Dictionary with git info
    """
    try:
        import subprocess
        
        # Get current commit hash
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('utf-8').strip()
        
        # Get branch name
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        ).decode('utf-8').strip()
        
        # Check for uncommitted changes
        status = subprocess.check_output(
            ['git', 'status', '--porcelain']
        ).decode('utf-8').strip()
        
        is_dirty = len(status) > 0
        
        return {
            'commit': commit,
            'branch': branch,
            'is_dirty': is_dirty
        }
    except:
        return {
            'commit': 'unknown',
            'branch': 'unknown',
            'is_dirty': False
        }


def print_section(title: str, char: str = "=", width: int = 70):
    """
    Print formatted section header
    
    Args:
        title: Section title
        char: Character to use for border
        width: Width of border
    """
    print("\n" + char * width)
    print(title.center(width))
    print(char * width)


def print_dict(d: Dict, indent: int = 0):
    """
    Pretty print nested dictionary
    
    Args:
        d: Dictionary to print
        indent: Indentation level
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_dict(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Recursively merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (overrides dict1)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Compute class weights for imbalanced data
    
    Args:
        labels: Class labels
        
    Returns:
        Class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    weights = total / (len(unique) * counts)
    
    return weights


def moving_average(data: List[float], window: int = 10) -> List[float]:
    """
    Compute moving average
    
    Args:
        data: List of values
        window: Window size
        
    Returns:
        Smoothed values
    """
    if len(data) < window:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(data[start:i+1]))
    
    return smoothed


def normalize_array(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize array
    
    Args:
        arr: Input array
        method: Normalization method ('minmax', 'zscore')
        
    Returns:
        Normalized array
    """
    if method == 'minmax':
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val > 0:
            return (arr - min_val) / (max_val - min_val)
        else:
            return arr
    elif method == 'zscore':
        mean = arr.mean()
        std = arr.std()
        if std > 0:
            return (arr - mean) / std
        else:
            return arr - mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping helper"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if should stop
        
        Args:
            value: Current metric value
            
        Returns:
            True if should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


if __name__ == "__main__":
    # Test utilities
    print("Testing helper utilities...")
    
    # Test timer
    with Timer("Test operation"):
        time.sleep(1)
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode='min')
    
    values = [1.0, 0.9, 0.8, 0.85, 0.87, 0.88]
    for i, val in enumerate(values):
        should_stop = early_stop(val)
        print(f"Epoch {i+1}: value={val:.2f}, stop={should_stop}")
        if should_stop:
            break
    
    # Test moving average
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    smoothed = moving_average(data, window=3)
    print(f"\nOriginal: {data}")
    print(f"Smoothed: {[f'{x:.1f}' for x in smoothed]}")
    
    print("\n✓ Helper utilities test complete!")
