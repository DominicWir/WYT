"""
Data Preprocessing Pipeline for Sensor Fusion

This module handles data loading, normalization, temporal alignment,
and dataset preparation for training the xLSTM model.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import json
import os


class SensorDataNormalizer:
    """
    Normalize and standardize sensor data.
    """
    
    def __init__(self):
        self.scalers = {
            'imu': StandardScaler(),
            'magnetic': StandardScaler(),
            'wifi': StandardScaler()
        }
        self.fitted = False
    
    def fit(
        self,
        imu_data: np.ndarray,
        mag_data: np.ndarray,
        wifi_data: Optional[np.ndarray] = None
    ):
        """
        Fit scalers on training data.
        
        Args:
            imu_data: IMU data of shape (n_samples, 6)
            mag_data: Magnetic data of shape (n_samples, 3)
            wifi_data: Optional Wi-Fi data of shape (n_samples, n_aps)
        """
        self.scalers['imu'].fit(imu_data)
        self.scalers['magnetic'].fit(mag_data)
        
        if wifi_data is not None:
            # Handle NaN values in Wi-Fi data
            wifi_clean = np.nan_to_num(wifi_data, nan=-100.0)
            self.scalers['wifi'].fit(wifi_clean)
        
        self.fitted = True
    
    def transform(
        self,
        imu_data: np.ndarray,
        mag_data: np.ndarray,
        wifi_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Transform sensor data using fitted scalers.
        
        Args:
            imu_data: IMU data
            mag_data: Magnetic data
            wifi_data: Optional Wi-Fi data
        
        Returns:
            Tuple of normalized (imu, magnetic, wifi) data
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        imu_norm = self.scalers['imu'].transform(imu_data)
        mag_norm = self.scalers['magnetic'].transform(mag_data)
        
        wifi_norm = None
        if wifi_data is not None:
            wifi_clean = np.nan_to_num(wifi_data, nan=-100.0)
            wifi_norm = self.scalers['wifi'].transform(wifi_clean)
        
        return imu_norm, mag_norm, wifi_norm
    
    def save(self, filepath: str):
        """Save normalizer parameters."""
        params = {
            'imu_mean': self.scalers['imu'].mean_.tolist(),
            'imu_scale': self.scalers['imu'].scale_.tolist(),
            'mag_mean': self.scalers['magnetic'].mean_.tolist(),
            'mag_scale': self.scalers['magnetic'].scale_.tolist(),
            'fitted': self.fitted
        }
        
        if hasattr(self.scalers['wifi'], 'mean_'):
            params['wifi_mean'] = self.scalers['wifi'].mean_.tolist()
            params['wifi_scale'] = self.scalers['wifi'].scale_.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
    
    def load(self, filepath: str):
        """Load normalizer parameters."""
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        self.scalers['imu'].mean_ = np.array(params['imu_mean'])
        self.scalers['imu'].scale_ = np.array(params['imu_scale'])
        self.scalers['magnetic'].mean_ = np.array(params['mag_mean'])
        self.scalers['magnetic'].scale_ = np.array(params['mag_scale'])
        
        if 'wifi_mean' in params:
            self.scalers['wifi'].mean_ = np.array(params['wifi_mean'])
            self.scalers['wifi'].scale_ = np.array(params['wifi_scale'])
        
        self.fitted = params['fitted']


class SensorSequenceDataset(Dataset):
    """
    PyTorch Dataset for sensor sequences.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        positions: np.ndarray,
        activities: np.ndarray,
        sequence_length: int = 20
    ):
        """
        Initialize dataset.
        
        Args:
            features: Feature array of shape (n_windows, feature_dim)
            positions: Position labels of shape (n_windows, 3)
            activities: Activity labels of shape (n_windows,)
            sequence_length: Length of sequences for training
        """
        self.features = torch.FloatTensor(features)
        self.positions = torch.FloatTensor(positions)
        self.activities = torch.LongTensor(activities)
        self.sequence_length = sequence_length
        
        # Calculate valid starting indices
        self.valid_indices = list(range(len(features) - sequence_length + 1))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        return {
            'features': self.features[start_idx:end_idx],
            'positions': self.positions[start_idx:end_idx],
            'activities': self.activities[start_idx:end_idx]
        }


def create_train_val_split(
    features: np.ndarray,
    positions: np.ndarray,
    activities: np.ndarray,
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[Dict, Dict]:
    """
    Split data into training and validation sets.
    
    Args:
        features: Feature array
        positions: Position labels
        activities: Activity labels
        val_ratio: Ratio of validation data
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data) dictionaries
    """
    np.random.seed(random_seed)
    
    n_samples = len(features)
    indices = np.random.permutation(n_samples)
    
    split_idx = int(n_samples * (1 - val_ratio))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_data = {
        'features': features[train_indices],
        'positions': positions[train_indices],
        'activities': activities[train_indices]
    }
    
    val_data = {
        'features': features[val_indices],
        'positions': positions[val_indices],
        'activities': activities[val_indices]
    }
    
    return train_data, val_data


def create_dataloaders(
    train_data: Dict,
    val_data: Dict,
    batch_size: int = 32,
    sequence_length: int = 20,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        batch_size: Batch size
        sequence_length: Sequence length
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = SensorSequenceDataset(
        train_data['features'],
        train_data['positions'],
        train_data['activities'],
        sequence_length
    )
    
    val_dataset = SensorSequenceDataset(
        val_data['features'],
        val_data['positions'],
        val_data['activities'],
        sequence_length
    )
    
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
    
    return train_loader, val_loader


def handle_missing_data(
    imu_data: np.ndarray,
    mag_data: np.ndarray,
    wifi_data: Optional[np.ndarray] = None,
    method: str = 'interpolate'
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Handle missing data in sensor streams.
    
    Args:
        imu_data: IMU data
        mag_data: Magnetic data
        wifi_data: Optional Wi-Fi data
        method: Method for handling missing data ('interpolate', 'forward_fill', 'zero')
    
    Returns:
        Tuple of cleaned sensor data
    """
    def clean_array(arr: np.ndarray, method: str) -> np.ndarray:
        if method == 'interpolate':
            # Linear interpolation for NaN values
            mask = np.isnan(arr)
            if mask.any():
                arr = arr.copy()
                for i in range(arr.shape[1]):
                    if mask[:, i].any():
                        valid_idx = ~mask[:, i]
                        if valid_idx.any():
                            arr[mask[:, i], i] = np.interp(
                                np.where(mask[:, i])[0],
                                np.where(valid_idx)[0],
                                arr[valid_idx, i]
                            )
        elif method == 'forward_fill':
            # Forward fill
            arr = arr.copy()
            for i in range(1, len(arr)):
                mask = np.isnan(arr[i])
                arr[i][mask] = arr[i-1][mask]
        elif method == 'zero':
            # Replace with zeros
            arr = np.nan_to_num(arr, nan=0.0)
        
        return arr
    
    imu_clean = clean_array(imu_data, method)
    mag_clean = clean_array(mag_data, method)
    wifi_clean = clean_array(wifi_data, method) if wifi_data is not None else None
    
    return imu_clean, mag_clean, wifi_clean


def remove_outliers(
    data: np.ndarray,
    threshold: float = 3.0
) -> np.ndarray:
    """
    Remove outliers using z-score method.
    
    Args:
        data: Input data
        threshold: Z-score threshold
    
    Returns:
        Data with outliers removed (replaced with median)
    """
    data_clean = data.copy()
    
    for i in range(data.shape[1]):
        col = data[:, i]
        mean = np.mean(col)
        std = np.std(col)
        
        if std > 0:
            z_scores = np.abs((col - mean) / std)
            outliers = z_scores > threshold
            
            if outliers.any():
                median = np.median(col[~outliers])
                data_clean[outliers, i] = median
    
    return data_clean


if __name__ == "__main__":
    print("=== Data Preprocessing Demo ===\n")
    
    # Create sample data
    n_samples = 1000
    imu_data = np.random.randn(n_samples, 6) * 2
    mag_data = np.random.randn(n_samples, 3) * 10 + 30
    wifi_data = np.random.randn(n_samples, 5) * 5 - 70
    
    # Add some missing values
    imu_data[50:60, 0] = np.nan
    wifi_data[100:110, :] = np.nan
    
    print(f"Original data shapes:")
    print(f"  IMU: {imu_data.shape}, NaN count: {np.isnan(imu_data).sum()}")
    print(f"  Magnetic: {mag_data.shape}, NaN count: {np.isnan(mag_data).sum()}")
    print(f"  Wi-Fi: {wifi_data.shape}, NaN count: {np.isnan(wifi_data).sum()}\n")
    
    # Handle missing data
    imu_clean, mag_clean, wifi_clean = handle_missing_data(
        imu_data, mag_data, wifi_data, method='interpolate'
    )
    
    print(f"After cleaning:")
    print(f"  IMU NaN count: {np.isnan(imu_clean).sum()}")
    print(f"  Magnetic NaN count: {np.isnan(mag_clean).sum()}")
    print(f"  Wi-Fi NaN count: {np.isnan(wifi_clean).sum()}\n")
    
    # Normalize data
    normalizer = SensorDataNormalizer()
    normalizer.fit(imu_clean, mag_clean, wifi_clean)
    
    imu_norm, mag_norm, wifi_norm = normalizer.transform(
        imu_clean, mag_clean, wifi_clean
    )
    
    print(f"Normalized data statistics:")
    print(f"  IMU mean: {imu_norm.mean():.3f}, std: {imu_norm.std():.3f}")
    print(f"  Magnetic mean: {mag_norm.mean():.3f}, std: {mag_norm.std():.3f}")
    print(f"  Wi-Fi mean: {wifi_norm.mean():.3f}, std: {wifi_norm.std():.3f}\n")
    
    # Create dummy labels
    positions = np.random.randn(n_samples, 3)
    activities = np.random.randint(0, 5, n_samples)
    
    # Create features (simplified)
    features = np.concatenate([imu_norm, mag_norm, wifi_norm], axis=1)
    
    # Split data
    train_data, val_data = create_train_val_split(
        features, positions, activities, val_ratio=0.2
    )
    
    print(f"Data split:")
    print(f"  Training samples: {len(train_data['features'])}")
    print(f"  Validation samples: {len(val_data['features'])}\n")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, batch_size=32, sequence_length=20
    )
    
    print(f"DataLoaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\nSample batch shapes:")
    print(f"  Features: {batch['features'].shape}")
    print(f"  Positions: {batch['positions'].shape}")
    print(f"  Activities: {batch['activities'].shape}")
