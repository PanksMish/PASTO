"""
XGBoost Baseline for Student Dropout Prediction
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from typing import Dict, Tuple
import pickle
from pathlib import Path


class XGBoostBaseline:
    """
    XGBoost baseline for dropout prediction
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize XGBoost baseline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default XGBoost parameters
        self.params = {
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'n_estimators': self.config.get('n_estimators', 100),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'min_child_weight': self.config.get('min_child_weight', 1),
            'gamma': self.config.get('gamma', 0),
            'reg_alpha': self.config.get('reg_alpha', 0),
            'reg_lambda': self.config.get('reg_lambda', 1),
            'random_state': 42,
            'use_label_encoder': False
        }
        
        # Handle class imbalance
        if self.config.get('scale_pos_weight', None):
            self.params['scale_pos_weight'] = self.config['scale_pos_weight']
        
        self.model = xgb.XGBClassifier(**self.params)
        
    def flatten_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Flatten temporal sequences to 2D for XGBoost
        
        Args:
            sequences: 3D sequences (N, T, F)
            
        Returns:
            2D features (N, T*F)
        """
        N, T, F = sequences.shape
        return sequences.reshape(N, T * F)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        tune_hyperparameters: bool = False
    ):
        """
        Train XGBoost model
        
        Args:
            X_train: Training sequences (N, T, F)
            y_train: Training labels (N,)
            X_val: Validation sequences
            y_val: Validation labels
            tune_hyperparameters: Whether to tune hyperparameters
        """
        # Flatten sequences
        X_train_flat = self.flatten_sequences(X_train)
        
        if tune_hyperparameters:
            print("Tuning hyperparameters with GridSearchCV...")
            self._tune_hyperparameters(X_train_flat, y_train)
        
        # Prepare validation set
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_flat = self.flatten_sequences(X_val)
            eval_set = [(X_val_flat, y_val)]
        
        # Train model
        print("Training XGBoost model...")
        self.model.fit(
            X_train_flat,
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        print("✓ XGBoost training complete")
        
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """Tune hyperparameters using GridSearchCV"""
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100, 200],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring=make_scorer(f1_score),
            cv=3,
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict dropout probabilities
        
        Args:
            X: Input sequences (N, T, F)
            
        Returns:
            Dropout probabilities (N,)
        """
        X_flat = self.flatten_sequences(X)
        probs = self.model.predict_proba(X_flat)[:, 1]
        return probs
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict dropout labels
        
        Args:
            X: Input sequences (N, T, F)
            threshold: Classification threshold
            
        Returns:
            Binary predictions (N,)
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        return self.model.feature_importances_
    
    def save(self, path: str):
        """Save model to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Model loaded from {path}")


def train_xgboost_baseline(
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    val_sequences: np.ndarray,
    val_labels: np.ndarray,
    config: Dict = None
) -> XGBoostBaseline:
    """
    Convenience function to train XGBoost baseline
    
    Args:
        train_sequences: Training sequences
        train_labels: Training labels
        val_sequences: Validation sequences
        val_labels: Validation labels
        config: Configuration
        
    Returns:
        Trained XGBoost model
    """
    # Calculate scale_pos_weight for imbalanced data
    if config is None:
        config = {}
    
    if 'scale_pos_weight' not in config:
        neg_count = np.sum(train_labels == 0)
        pos_count = np.sum(train_labels == 1)
        config['scale_pos_weight'] = neg_count / pos_count if pos_count > 0 else 1
        print(f"Calculated scale_pos_weight: {config['scale_pos_weight']:.2f}")
    
    # Initialize and train
    model = XGBoostBaseline(config)
    model.fit(train_sequences, train_labels, val_sequences, val_labels)
    
    return model


if __name__ == "__main__":
    # Test XGBoost baseline
    print("Testing XGBoost Baseline...")
    
    # Create dummy data
    N_train, N_val = 1000, 200
    T, F = 30, 10
    
    X_train = np.random.randn(N_train, T, F)
    y_train = np.random.randint(0, 2, N_train)
    
    X_val = np.random.randn(N_val, T, F)
    y_val = np.random.randint(0, 2, N_val)
    
    # Train model
    model = train_xgboost_baseline(X_train, y_train, X_val, y_val)
    
    # Test predictions
    probs = model.predict_proba(X_val)
    preds = model.predict(X_val)
    
    print(f"\nPredictions shape: {preds.shape}")
    print(f"Probabilities range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"Positive predictions: {preds.sum()}")
    
    # Feature importance
    importance = model.get_feature_importance()
    print(f"\nFeature importance shape: {importance.shape}")
    print(f"Top 5 features: {np.argsort(importance)[-5:][::-1]}")
    
    print("\n✓ XGBoost baseline test complete!")
