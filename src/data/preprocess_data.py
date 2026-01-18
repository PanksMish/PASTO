"""
Data Preprocessing Module for PASTO
Handles feature engineering, normalization, SMOTE, and data augmentation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering for temporal student data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scalers = {}
        self.encoders = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw data
        
        Args:
            df: Raw temporal dataframe
            
        Returns:
            DataFrame with engineered features
        """
        print("Engineering features...")
        
        df = df.copy()
        
        # Temporal aggregation features
        df = self._create_temporal_aggregations(df)
        
        # Trend features
        df = self._create_trend_features(df)
        
        # Engagement features
        df = self._create_engagement_features(df)
        
        # Performance features
        df = self._create_performance_features(df)
        
        # Behavioral change features
        df = self._create_behavioral_change_features(df)
        
        print(f"✓ Engineered {len(df.columns)} features")
        
        return df
    
    def _create_temporal_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal aggregation features"""
        
        # Sort by student and week
        df = df.sort_values(['id_student', 'week'])
        
        # Rolling statistics (3-week window)
        rolling_cols = ['sum_click', 'avg_score', 'days_active']
        for col in rolling_cols:
            if col in df.columns:
                df[f'{col}_rolling_mean_3w'] = df.groupby('id_student')[col].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )
                df[f'{col}_rolling_std_3w'] = df.groupby('id_student')[col].transform(
                    lambda x: x.rolling(window=3, min_periods=1).std()
                )
        
        # Cumulative features
        for col in rolling_cols:
            if col in df.columns:
                df[f'{col}_cumsum'] = df.groupby('id_student')[col].cumsum()
                df[f'{col}_cummean'] = df.groupby('id_student')[col].transform(
                    lambda x: x.expanding().mean()
                )
        
        return df
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend and momentum features"""
        
        # Week-over-week changes
        for col in ['sum_click', 'avg_score']:
            if col in df.columns:
                df[f'{col}_diff'] = df.groupby('id_student')[col].diff()
                df[f'{col}_pct_change'] = df.groupby('id_student')[col].pct_change()
        
        # Momentum (recent vs historical average)
        for col in ['sum_click', 'avg_score']:
            if col in df.columns:
                recent_mean = df.groupby('id_student')[col].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )
                historical_mean = df.groupby('id_student')[col].transform(
                    lambda x: x.expanding().mean()
                )
                df[f'{col}_momentum'] = recent_mean - historical_mean
        
        return df
    
    def _create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engagement-related features"""
        
        # Engagement rate
        if 'sum_click' in df.columns and 'days_active' in df.columns:
            df['engagement_rate'] = df['sum_click'] / (df['days_active'] + 1)
        
        # Activity consistency (coefficient of variation)
        if 'sum_click' in df.columns:
            df['activity_consistency'] = df.groupby('id_student')['sum_click'].transform(
                lambda x: x.std() / (x.mean() + 1)
            )
        
        # Days since last activity
        if 'days_active' in df.columns:
            df['days_since_last_active'] = df.groupby('id_student')['week'].diff()
            df.loc[df['days_active'] > 0, 'days_since_last_active'] = 0
        
        # Engagement decay (exponentially weighted)
        if 'sum_click' in df.columns:
            df['engagement_decay'] = df.groupby('id_student')['sum_click'].transform(
                lambda x: x.ewm(alpha=0.3).mean()
            )
        
        return df
    
    def _create_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create academic performance features"""
        
        # Grade trends
        if 'avg_score' in df.columns:
            # Best and worst scores
            df['best_score'] = df.groupby('id_student')['avg_score'].transform('max')
            df['worst_score'] = df.groupby('id_student')['avg_score'].transform('min')
            df['score_range'] = df['best_score'] - df['worst_score']
            
            # Performance consistency
            df['score_consistency'] = df.groupby('id_student')['avg_score'].transform(
                lambda x: 1 / (x.std() + 1)
            )
            
            # Improvement rate
            df['score_improvement'] = df.groupby('id_student')['avg_score'].transform(
                lambda x: (x.iloc[-1] - x.iloc[0]) / (len(x) + 1) if len(x) > 1 else 0
            )
        
        # Assessment completion rate
        if 'num_assessments' in df.columns:
            total_assessments = df.groupby('id_student')['num_assessments'].transform('sum')
            df['assessment_completion_rate'] = df['num_assessments'] / (total_assessments + 1)
        
        return df
    
    def _create_behavioral_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features capturing behavioral changes"""
        
        # Sudden drops in engagement
        if 'sum_click' in df.columns:
            df['engagement_drop'] = df.groupby('id_student')['sum_click'].transform(
                lambda x: (x < 0.5 * x.rolling(window=5, min_periods=1).mean()).astype(int)
            )
        
        # Irregular patterns (entropy of activity distribution)
        if 'sum_click' in df.columns:
            df['activity_entropy'] = df.groupby('id_student')['sum_click'].transform(
                lambda x: -np.sum(x * np.log(x + 1e-10)) if len(x) > 0 else 0
            )
        
        return df


class DataNormalizer:
    """
    Normalize features using various scaling methods
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize normalizer
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.scaler = None
        self._initialize_scaler()
        
    def _initialize_scaler(self):
        """Initialize the appropriate scaler"""
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def fit(self, sequences: np.ndarray) -> 'DataNormalizer':
        """
        Fit scaler on training data
        
        Args:
            sequences: Training sequences (N, T, F)
            
        Returns:
            Self
        """
        # Reshape to 2D for fitting
        N, T, F = sequences.shape
        sequences_2d = sequences.reshape(-1, F)
        
        self.scaler.fit(sequences_2d)
        
        return self
    
    def transform(self, sequences: np.ndarray) -> np.ndarray:
        """
        Transform sequences using fitted scaler
        
        Args:
            sequences: Sequences to transform (N, T, F)
            
        Returns:
            Normalized sequences
        """
        N, T, F = sequences.shape
        sequences_2d = sequences.reshape(-1, F)
        
        normalized_2d = self.scaler.transform(sequences_2d)
        normalized = normalized_2d.reshape(N, T, F)
        
        return normalized
    
    def fit_transform(self, sequences: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(sequences).transform(sequences)


class ImbalanceHandler:
    """
    Handle class imbalance using SMOTE and variants
    """
    
    def __init__(
        self,
        method: str = 'smote',
        sampling_strategy: float = 0.5,
        k_neighbors: int = 5,
        random_state: int = 42
    ):
        """
        Initialize imbalance handler
        
        Args:
            method: Sampling method ('smote', 'adasyn', 'smoteenn', 'smotetomek')
            sampling_strategy: Target ratio of minority to majority class
            k_neighbors: Number of nearest neighbors
            random_state: Random seed
        """
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampler = None
        self._initialize_sampler()
    
    def _initialize_sampler(self):
        """Initialize the appropriate sampler"""
        if self.method == 'smote':
            self.sampler = SMOTE(
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state
            )
        elif self.method == 'adasyn':
            self.sampler = ADASYN(
                sampling_strategy=self.sampling_strategy,
                n_neighbors=self.k_neighbors,
                random_state=self.random_state
            )
        elif self.method == 'smoteenn':
            self.sampler = SMOTEENN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        elif self.method == 'smotetomek':
            self.sampler = SMOTETomek(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")
    
    def resample(
        self,
        sequences: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample sequences to handle class imbalance
        
        Args:
            sequences: Input sequences (N, T, F)
            labels: Binary labels (N,)
            
        Returns:
            Tuple of (resampled_sequences, resampled_labels)
        """
        print(f"Applying {self.method} resampling...")
        print(f"  Original distribution: {np.bincount(labels)}")
        
        # Flatten sequences for SMOTE (requires 2D input)
        N, T, F = sequences.shape
        sequences_2d = sequences.reshape(N, T * F)
        
        # Resample
        sequences_resampled, labels_resampled = self.sampler.fit_resample(
            sequences_2d, labels
        )
        
        # Reshape back to 3D
        N_new = len(sequences_resampled)
        sequences_resampled = sequences_resampled.reshape(N_new, T, F)
        
        print(f"  Resampled distribution: {np.bincount(labels_resampled)}")
        print(f"  New sample count: {N_new}")
        
        return sequences_resampled, labels_resampled


class TemporalAugmenter:
    """
    Augment temporal sequences with realistic perturbations
    """
    
    def __init__(self, config: Dict):
        """
        Initialize temporal augmenter
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def augment(
        self,
        sequences: np.ndarray,
        augmentation_factor: float = 0.2
    ) -> np.ndarray:
        """
        Augment temporal sequences
        
        Args:
            sequences: Input sequences (N, T, F)
            augmentation_factor: Fraction of samples to augment
            
        Returns:
            Augmented sequences
        """
        N, T, F = sequences.shape
        num_augment = int(N * augmentation_factor)
        
        # Select random sequences to augment
        indices = np.random.choice(N, size=num_augment, replace=False)
        augmented = sequences[indices].copy()
        
        # Apply augmentation techniques
        augmented = self._add_noise(augmented)
        augmented = self._time_warping(augmented)
        augmented = self._magnitude_warping(augmented)
        
        # Concatenate with original
        sequences_augmented = np.vstack([sequences, augmented])
        
        return sequences_augmented
    
    def _add_noise(self, sequences: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, sequences.shape)
        return sequences + noise
    
    def _time_warping(self, sequences: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Apply time warping"""
        # Simple implementation: duplicate or drop random time steps
        N, T, F = sequences.shape
        warped = sequences.copy()
        
        for i in range(N):
            if np.random.rand() < 0.5:
                # Duplicate a random time step
                idx = np.random.randint(0, T-1)
                warped[i, idx+1:] = sequences[i, idx:-1]
            else:
                # Drop a random time step
                idx = np.random.randint(0, T-1)
                warped[i, idx:-1] = sequences[i, idx+1:]
                warped[i, -1] = sequences[i, -1]
        
        return warped
    
    def _magnitude_warping(self, sequences: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """Apply magnitude warping"""
        N, T, F = sequences.shape
        
        # Generate smooth random curves
        knot = np.random.normal(loc=1.0, scale=sigma, size=(N, F))
        warping = np.tile(knot[:, np.newaxis, :], (1, T, 1))
        
        return sequences * warping


def preprocess_pipeline(
    sequences: np.ndarray,
    dropout_labels: np.ndarray,
    trajectory_labels: np.ndarray,
    config: Dict,
    is_training: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline
    
    Args:
        sequences: Raw sequences (N, T, F)
        dropout_labels: Dropout labels (N,)
        trajectory_labels: Trajectory labels (N,)
        config: Configuration dictionary
        is_training: Whether this is training data
        
    Returns:
        Tuple of (processed_sequences, dropout_labels, trajectory_labels)
    """
    print("\n" + "="*50)
    print("PREPROCESSING PIPELINE")
    print("="*50)
    
    # 1. Normalization
    if config.get('normalize', True):
        print("\n[1/3] Normalizing features...")
        normalizer = DataNormalizer(method=config.get('normalization_method', 'standard'))
        
        if is_training:
            sequences = normalizer.fit_transform(sequences)
        else:
            sequences = normalizer.transform(sequences)
        
        print(f"  ✓ Applied {config.get('normalization_method', 'standard')} normalization")
    
    # 2. Handle class imbalance (only for training)
    if is_training and config.get('use_smote', False):
        print("\n[2/3] Handling class imbalance...")
        imbalance_handler = ImbalanceHandler(
            method='smote',
            sampling_strategy=config.get('sampling_strategy', 0.5),
            k_neighbors=config.get('smote_k_neighbors', 5)
        )
        
        sequences, dropout_labels = imbalance_handler.resample(sequences, dropout_labels)
        
        # Duplicate trajectory labels to match
        trajectory_labels = np.tile(trajectory_labels, len(sequences) // len(trajectory_labels) + 1)
        trajectory_labels = trajectory_labels[:len(sequences)]
    
    # 3. Data augmentation (only for training)
    if is_training and config.get('use_augmentation', False):
        print("\n[3/3] Augmenting data...")
        augmenter = TemporalAugmenter(config)
        sequences = augmenter.augment(sequences, augmentation_factor=0.2)
        
        # Duplicate labels to match
        n_original = len(dropout_labels)
        n_augmented = len(sequences) - n_original
        
        dropout_labels = np.concatenate([
            dropout_labels,
            dropout_labels[:n_augmented]
        ])
        
        trajectory_labels = np.concatenate([
            trajectory_labels,
            trajectory_labels[:n_augmented]
        ])
        
        print(f"  ✓ Augmented dataset: {len(sequences)} samples")
    
    print("\n" + "="*50)
    print(f"PREPROCESSING COMPLETE")
    print(f"  Final dataset size: {len(sequences)}")
    print(f"  Sequence shape: {sequences.shape}")
    print(f"  Dropout rate: {dropout_labels.mean():.2%}")
    print("="*50 + "\n")
    
    return sequences, dropout_labels, trajectory_labels


if __name__ == "__main__":
    # Example usage
    from dataset_loader import OULADDatasetLoader
    
    # Load data
    loader = OULADDatasetLoader("data/raw/oulad", sequence_length=30)
    temporal_df = loader.create_temporal_features()
    
    # Engineer features
    engineer = FeatureEngineer(config={})
    temporal_df = engineer.engineer_features(temporal_df)
    
    # Create sequences
    feature_cols = [col for col in temporal_df.columns 
                    if col not in ['id_student', 'code_module', 'code_presentation', 
                                   'week', 'dropout', 'final_result']]
    
    sequences, dropout_labels, trajectory_labels, _ = loader.create_sequences(
        temporal_df, feature_cols[:10]  # Use first 10 features
    )
    
    # Preprocess
    config = {
        'normalize': True,
        'normalization_method': 'standard',
        'use_smote': True,
        'sampling_strategy': 0.5,
        'smote_k_neighbors': 5
    }
    
    sequences, dropout_labels, trajectory_labels = preprocess_pipeline(
        sequences, dropout_labels, trajectory_labels, config, is_training=True
    )
