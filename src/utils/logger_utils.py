"""
Logging utilities for PASTO
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "pasto",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_file: Log file name (if None, auto-generate)
        level: Logging level
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'pasto_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """
    Logger for tracking training metrics
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize metrics logger
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / 'metrics.csv'
        
        # Initialize CSV file
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w') as f:
                f.write('epoch,split,metric,value\n')
    
    def log_metric(
        self,
        epoch: int,
        split: str,
        metric: str,
        value: float
    ):
        """
        Log a single metric
        
        Args:
            epoch: Epoch number
            split: Data split (train/val/test)
            metric: Metric name
            value: Metric value
        """
        with open(self.metrics_file, 'a') as f:
            f.write(f'{epoch},{split},{metric},{value}\n')
    
    def log_metrics_dict(
        self,
        epoch: int,
        split: str,
        metrics: dict
    ):
        """
        Log multiple metrics
        
        Args:
            epoch: Epoch number
            split: Data split
            metrics: Dictionary of metrics
        """
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_metric(epoch, split, metric, value)


if __name__ == "__main__":
    # Test logger
    logger = setup_logger(
        name="test",
        log_dir="logs/test",
        console=True
    )
    
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Test metrics logger
    metrics_logger = MetricsLogger("logs/test")
    metrics_logger.log_metric(1, "train", "loss", 0.5)
    metrics_logger.log_metrics_dict(1, "val", {
        "accuracy": 0.85,
        "f1": 0.82
    })
    
    print("✓ Logger test complete!")
