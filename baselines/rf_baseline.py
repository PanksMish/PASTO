"""
Random Forest Baseline for Student Dropout Prediction
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from typing import Dict
import pickle
from pathlib import Path


class RandomForestBaseline:
    """
    Random Forest baseline for dropout prediction
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Random Forest baseline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default Random Forest parameters
        self.params = {
            'n_estimators': self.config.get('n_estimators', 100),
            'max_depth': self.config.get('max_depth', None),
            'min_samples_split': self.config.get('min_samples_split', 2),
            'min_samples_leaf': self.config.get('min_samples_leaf', 1),
            'max_features': self.config.get('max_features', 'sqrt'),
            'bootstrap': self.config.get('bootstrap', True),
            'class_weight': self.config.get('class_weight', 'balanced'),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }
        
        self.model = RandomForestClassifier(**self.params)
        
    def flatten_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Flatten temporal sequences to 2D
        
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
        tune_hyperparameters: bool = False
    ):
        """
        Train Random Forest model
        
        Args:
            X_train: Training sequences (N, T, F)
            y_train: Training labels (N,)
            tune_hyperparameters: Whether to tune hyperparameters
        """
        # Flatten sequences
        X_train_flat = self.flatten_sequences(X_train)
        
        if tune_hyperparameters:
            print("Tuning hyperparameters with GridSearchCV...")
            self._tune_hyperparameters(X_train_flat, y_train)
        
        # Train model
        print("Training Random Forest model...")
        self.model.fit(X_train_flat, y_train)
        
        print("✓ Random Forest training complete")
        
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """Tune hyperparameters using GridSearchCV"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
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


if __name__ == "__main__":
    # Test Random Forest baseline
    print("Testing Random Forest Baseline...")
    
    # Create dummy data
    N = 1000
    T, F = 30, 10
    
    X = np.random.randn(N, T, F)
    y = np.random.randint(0, 2, N)
    
    # Train model
    model = RandomForestBaseline()
    model.fit(X, y)
    
    # Test predictions
    probs = model.predict_proba(X)
    preds = model.predict(X)
    
    print(f"\nPredictions shape: {preds.shape}")
    print(f"Probabilities range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    print("\n✓ Random Forest baseline test complete!")
