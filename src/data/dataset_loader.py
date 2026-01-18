"""
Dataset Loader for PASTO Framework
Handles loading and preprocessing of educational datasets (OULAD, Indian School, etc.)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class OULADDatasetLoader:
    """
    Loader for Open University Learning Analytics Dataset (OULAD)
    
    Dataset contains:
    - studentInfo.csv: Student demographic and registration info
    - studentAssessment.csv: Assessment results
    - studentVle.csv: Virtual Learning Environment interactions
    - courses.csv: Course information
    - assessments.csv: Assessment metadata
    - vle.csv: VLE activity metadata
    """
    
    def __init__(self, data_dir: str, sequence_length: int = 30):
        """
        Initialize OULAD dataset loader
        
        Args:
            data_dir: Path to directory containing OULAD CSV files
            sequence_length: Number of time steps in each sequence
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        
        # Verify data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Load all tables
        self.student_info = None
        self.student_assessment = None
        self.student_vle = None
        self.courses = None
        self.assessments = None
        self.vle = None
        
        self._load_tables()
        
    def _load_tables(self):
        """Load all OULAD CSV tables"""
        print("Loading OULAD dataset tables...")
        
        try:
            self.student_info = pd.read_csv(self.data_dir / 'studentInfo.csv')
            self.student_assessment = pd.read_csv(self.data_dir / 'studentAssessment.csv')
            self.student_vle = pd.read_csv(self.data_dir / 'studentVle.csv')
            self.courses = pd.read_csv(self.data_dir / 'courses.csv')
            self.assessments = pd.read_csv(self.data_dir / 'assessments.csv')
            self.vle = pd.read_csv(self.data_dir / 'vle.csv')
            
            print(f"✓ Loaded {len(self.student_info)} students")
            print(f"✓ Loaded {len(self.student_vle)} VLE interactions")
            print(f"✓ Loaded {len(self.student_assessment)} assessment records")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing OULAD file: {e}")
    
    def create_temporal_features(self) -> pd.DataFrame:
        """
        Create temporal feature sequences for each student
        
        Returns:
            DataFrame with temporal features aggregated by week
        """
        print("\nCreating temporal features...")
        
        # Create weekly aggregations of VLE activity
        vle_weekly = self._aggregate_vle_weekly()
        
        # Create assessment features
        assessment_features = self._create_assessment_features()
        
        # Merge student info with temporal features
        temporal_df = self._merge_temporal_data(vle_weekly, assessment_features)
        
        return temporal_df
    
    def _aggregate_vle_weekly(self) -> pd.DataFrame:
        """Aggregate VLE interactions by week"""
        # Convert date to week number
        self.student_vle['week'] = self.student_vle['date'] // 7
        
        # Aggregate by student, course, and week
        vle_agg = self.student_vle.groupby(
            ['code_module', 'code_presentation', 'id_student', 'week']
        ).agg({
            'sum_click': 'sum',
            'date': 'count'  # Number of days active
        }).rename(columns={'date': 'days_active'}).reset_index()
        
        return vle_agg
    
    def _create_assessment_features(self) -> pd.DataFrame:
        """Create assessment-based features"""
        # Merge assessment metadata
        assess_with_meta = self.student_assessment.merge(
            self.assessments,
            on=['id_assessment', 'code_module', 'code_presentation'],
            how='left'
        )
        
        # Convert date to week
        assess_with_meta['week'] = assess_with_meta['date'] // 7
        
        # Aggregate by student and week
        assess_agg = assess_with_meta.groupby(
            ['code_module', 'code_presentation', 'id_student', 'week']
        ).agg({
            'score': ['mean', 'std', 'count'],
            'weight': 'sum'
        }).reset_index()
        
        assess_agg.columns = [
            'code_module', 'code_presentation', 'id_student', 'week',
            'avg_score', 'std_score', 'num_assessments', 'total_weight'
        ]
        
        return assess_agg
    
    def _merge_temporal_data(
        self, 
        vle_weekly: pd.DataFrame, 
        assessment_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge all temporal features with student info"""
        
        # Merge VLE and assessment features
        temporal = vle_weekly.merge(
            assessment_features,
            on=['code_module', 'code_presentation', 'id_student', 'week'],
            how='outer'
        )
        
        # Merge with student info
        temporal = temporal.merge(
            self.student_info,
            on=['code_module', 'code_presentation', 'id_student'],
            how='left'
        )
        
        # Fill missing values
        temporal['sum_click'] = temporal['sum_click'].fillna(0)
        temporal['days_active'] = temporal['days_active'].fillna(0)
        temporal['avg_score'] = temporal['avg_score'].fillna(
            temporal.groupby('id_student')['avg_score'].transform('mean')
        )
        temporal['num_assessments'] = temporal['num_assessments'].fillna(0)
        
        # Create dropout label (1 if Withdrawn or Fail, 0 otherwise)
        temporal['dropout'] = temporal['final_result'].isin(['Withdrawn', 'Fail']).astype(int)
        
        # Sort by student and week
        temporal = temporal.sort_values(['id_student', 'week'])
        
        return temporal
    
    def create_sequences(
        self, 
        temporal_df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create fixed-length sequences for each student
        
        Args:
            temporal_df: DataFrame with temporal features
            feature_columns: List of feature column names
            
        Returns:
            Tuple of (sequences, dropout_labels, trajectory_labels, student_ids)
        """
        print(f"\nCreating sequences of length {self.sequence_length}...")
        
        sequences = []
        dropout_labels = []
        trajectory_labels = []
        student_ids = []
        
        # Group by student
        for student_id, group in temporal_df.groupby('id_student'):
            group = group.sort_values('week')
            
            # Get feature values
            features = group[feature_columns].values
            
            # Get labels
            dropout_label = group['dropout'].iloc[-1]  # Final dropout status
            
            # Trajectory label: final score or engagement metric
            if 'avg_score' in group.columns and not group['avg_score'].isna().all():
                trajectory_label = group['avg_score'].mean()
            else:
                trajectory_label = 0.0
            
            # Create sequences with padding/truncation
            if len(features) < self.sequence_length:
                # Pad with zeros
                pad_length = self.sequence_length - len(features)
                features = np.vstack([
                    np.zeros((pad_length, len(feature_columns))),
                    features
                ])
            else:
                # Take last sequence_length steps
                features = features[-self.sequence_length:]
            
            sequences.append(features)
            dropout_labels.append(dropout_label)
            trajectory_labels.append(trajectory_label)
            student_ids.append(student_id)
        
        sequences = np.array(sequences, dtype=np.float32)
        dropout_labels = np.array(dropout_labels, dtype=np.int64)
        trajectory_labels = np.array(trajectory_labels, dtype=np.float32)
        student_ids = np.array(student_ids)
        
        print(f"✓ Created {len(sequences)} sequences")
        print(f"  Sequence shape: {sequences.shape}")
        print(f"  Dropout rate: {dropout_labels.mean():.2%}")
        
        return sequences, dropout_labels, trajectory_labels, student_ids


class StudentDropoutDataset(Dataset):
    """
    PyTorch Dataset for student dropout prediction
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        dropout_labels: np.ndarray,
        trajectory_labels: np.ndarray,
        student_ids: np.ndarray,
        demographics: Optional[pd.DataFrame] = None,
        transform=None
    ):
        """
        Initialize dataset
        
        Args:
            sequences: Temporal feature sequences (N, T, F)
            dropout_labels: Binary dropout labels (N,)
            trajectory_labels: Continuous trajectory outcomes (N,)
            student_ids: Student identifiers (N,)
            demographics: Optional demographic features
            transform: Optional transform to apply
        """
        self.sequences = torch.FloatTensor(sequences)
        self.dropout_labels = torch.LongTensor(dropout_labels)
        self.trajectory_labels = torch.FloatTensor(trajectory_labels)
        self.student_ids = student_ids
        self.demographics = demographics
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        dropout_label = self.dropout_labels[idx]
        trajectory_label = self.trajectory_labels[idx]
        student_id = self.student_ids[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        item = {
            'sequence': sequence,
            'dropout_label': dropout_label,
            'trajectory_label': trajectory_label,
            'student_id': student_id
        }
        
        if self.demographics is not None:
            # Add demographic features
            demo_features = self.demographics.loc[
                self.demographics['id_student'] == student_id
            ].iloc[0]
            item['demographics'] = torch.FloatTensor(demo_features.values)
        
        return item


class IndianSchoolDatasetLoader:
    """
    Loader for Indian School Dataset
    Annual academic records with multi-year trajectories
    """
    
    def __init__(self, data_path: str):
        """
        Initialize Indian School dataset loader
        
        Args:
            data_path: Path to Indian school dataset file
        """
        self.data_path = Path(data_path)
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """Load Indian school dataset"""
        print(f"Loading Indian School dataset from {self.data_path}...")
        
        if self.data_path.suffix == '.csv':
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.suffix in ['.xlsx', '.xls']:
            self.data = pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        print(f"✓ Loaded {len(self.data)} records")
    
    def preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess Indian school data
        
        Returns:
            Tuple of (sequences, dropout_labels, trajectory_labels)
        """
        # This is a placeholder - actual implementation depends on data format
        # Typically: year-wise grades, attendance, socio-economic features
        
        # Example processing (modify based on actual data schema)
        sequences = []
        dropout_labels = []
        trajectory_labels = []
        
        for student_id, group in self.data.groupby('student_id'):
            # Extract temporal features
            features = group[['grade', 'attendance', 'exam_score']].values
            sequences.append(features)
            
            # Extract labels
            dropout = group['dropout'].iloc[-1]
            trajectory = group['final_grade'].iloc[-1]
            
            dropout_labels.append(dropout)
            trajectory_labels.append(trajectory)
        
        return (
            np.array(sequences),
            np.array(dropout_labels),
            np.array(trajectory_labels)
        )


def create_dataloaders(
    sequences: np.ndarray,
    dropout_labels: np.ndarray,
    trajectory_labels: np.ndarray,
    student_ids: np.ndarray,
    demographics: Optional[pd.DataFrame] = None,
    batch_size: int = 256,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with stratified split
    
    Args:
        sequences: Temporal sequences
        dropout_labels: Dropout labels
        trajectory_labels: Trajectory labels
        student_ids: Student IDs
        demographics: Optional demographics
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        num_workers: Number of data loading workers
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        np.arange(len(sequences)),
        test_size=(val_ratio + test_ratio),
        stratify=dropout_labels,
        random_state=seed
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size),
        stratify=dropout_labels[temp_idx],
        random_state=seed
    )
    
    # Create datasets
    train_dataset = StudentDropoutDataset(
        sequences[train_idx],
        dropout_labels[train_idx],
        trajectory_labels[train_idx],
        student_ids[train_idx],
        demographics
    )
    
    val_dataset = StudentDropoutDataset(
        sequences[val_idx],
        dropout_labels[val_idx],
        trajectory_labels[val_idx],
        student_ids[val_idx],
        demographics
    )
    
    test_dataset = StudentDropoutDataset(
        sequences[test_idx],
        dropout_labels[test_idx],
        trajectory_labels[test_idx],
        student_ids[test_idx],
        demographics
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    loader = OULADDatasetLoader(
        data_dir="data/raw/oulad",
        sequence_length=30
    )
    
    # Create temporal features
    temporal_df = loader.create_temporal_features()
    
    # Define feature columns
    feature_columns = [
        'sum_click', 'days_active', 'avg_score',
        'num_assessments', 'studied_credits'
    ]
    
    # Create sequences
    sequences, dropout_labels, trajectory_labels, student_ids = \
        loader.create_sequences(temporal_df, feature_columns)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        sequences, dropout_labels, trajectory_labels, student_ids,
        batch_size=256
    )
    
    # Test iteration
    for batch in train_loader:
        print("\nBatch example:")
        print(f"  Sequences: {batch['sequence'].shape}")
        print(f"  Dropout labels: {batch['dropout_label'].shape}")
        print(f"  Trajectory labels: {batch['trajectory_label'].shape}")
        break
