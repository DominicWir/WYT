"""
Statistical Feature Extraction for Heterogeneous Sensor Data

This module extracts statistical features from IMU, Magnetic, and Wi-Fi sensor data
using sliding windows. Features include min, max, mean, std, and percentiles.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


class SensorFeatureExtractor:
    """
    Extract statistical features from heterogeneous sensor data streams.
    
    Supports:
    - IMU data (accelerometer: 3-axis, gyroscope: 3-axis)
    - Magnetic field data (3-axis)
    - Wi-Fi RSSI data (variable number of access points)
    """
    
    def __init__(
        self,
        window_size: int = 50,
        stride: int = 25,
        percentiles: List[float] = [25.0, 50.0, 75.0]
    ):
        """
        Initialize the feature extractor.
        
        Args:
            window_size: Number of samples in each window (default: 50 samples)
            stride: Number of samples to slide the window (default: 25 for 50% overlap)
            percentiles: List of percentiles to compute (default: [25, 50, 75])
        """
        self.window_size = window_size
        self.stride = stride
        self.percentiles = percentiles
        
        # Feature names for documentation
        self.feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive names for all features."""
        stats = ['min', 'max', 'mean', 'std'] + [f'p{int(p)}' for p in self.percentiles]
        
        feature_names = []
        
        # IMU features (6 axes: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
        imu_axes = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        for axis in imu_axes:
            for stat in stats:
                feature_names.append(f'imu_{axis}_{stat}')
        
        # Magnetic features (3 axes: mag_x, mag_y, mag_z)
        mag_axes = ['mag_x', 'mag_y', 'mag_z']
        for axis in mag_axes:
            for stat in stats:
                feature_names.append(f'mag_{axis}_{stat}')
        
        # Wi-Fi features (aggregated across APs)
        for stat in stats:
            feature_names.append(f'wifi_rssi_{stat}')
        
        # Additional derived features
        feature_names.extend([
            'imu_acc_magnitude_mean',
            'imu_acc_magnitude_std',
            'imu_gyro_magnitude_mean',
            'imu_gyro_magnitude_std',
            'mag_magnitude_mean',
            'mag_magnitude_std',
            'wifi_ap_count'
        ])
        
        return feature_names
    
    def extract_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from a single window of data.
        
        Args:
            data: Array of shape (window_size, n_features)
        
        Returns:
            Array of statistical features
        """
        if len(data) == 0:
            warnings.warn("Empty data window, returning zeros")
            return np.zeros(len(self.percentiles) + 4)
        
        features = []
        
        # Basic statistics
        features.append(np.min(data, axis=0))
        features.append(np.max(data, axis=0))
        features.append(np.mean(data, axis=0))
        features.append(np.std(data, axis=0))
        
        # Percentiles
        for p in self.percentiles:
            features.append(np.percentile(data, p, axis=0))
        
        # Concatenate all features
        return np.concatenate([f.flatten() if f.ndim > 0 else [f] for f in features])
    
    def extract_imu_features(self, imu_data: np.ndarray) -> np.ndarray:
        """
        Extract features from IMU data (accelerometer + gyroscope).
        
        Args:
            imu_data: Array of shape (window_size, 6) 
                     [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        
        Returns:
            Array of IMU features
        """
        if imu_data.shape[1] != 6:
            raise ValueError(f"Expected 6 IMU channels, got {imu_data.shape[1]}")
        
        # Statistical features for each axis
        stat_features = self.extract_statistical_features(imu_data)
        
        # Magnitude features
        acc_magnitude = np.linalg.norm(imu_data[:, :3], axis=1)
        gyro_magnitude = np.linalg.norm(imu_data[:, 3:], axis=1)
        
        magnitude_features = np.array([
            np.mean(acc_magnitude),
            np.std(acc_magnitude),
            np.mean(gyro_magnitude),
            np.std(gyro_magnitude)
        ])
        
        return np.concatenate([stat_features, magnitude_features])
    
    def extract_magnetic_features(self, mag_data: np.ndarray) -> np.ndarray:
        """
        Extract features from magnetic field data.
        
        Args:
            mag_data: Array of shape (window_size, 3) [mag_x, mag_y, mag_z]
        
        Returns:
            Array of magnetic features
        """
        if mag_data.shape[1] != 3:
            raise ValueError(f"Expected 3 magnetic channels, got {mag_data.shape[1]}")
        
        # Statistical features for each axis
        stat_features = self.extract_statistical_features(mag_data)
        
        # Magnitude features
        mag_magnitude = np.linalg.norm(mag_data, axis=1)
        magnitude_features = np.array([
            np.mean(mag_magnitude),
            np.std(mag_magnitude)
        ])
        
        return np.concatenate([stat_features, magnitude_features])
    
    def extract_wifi_features(self, wifi_data: np.ndarray) -> np.ndarray:
        """
        Extract features from Wi-Fi RSSI data.
        
        Args:
            wifi_data: Array of shape (window_size, n_access_points)
                      RSSI values for each access point
        
        Returns:
            Array of Wi-Fi features
        """
        if len(wifi_data) == 0 or wifi_data.shape[1] == 0:
            # No Wi-Fi data available
            return np.zeros(len(self.percentiles) + 5)
        
        # Flatten all RSSI values across APs for aggregate statistics
        all_rssi = wifi_data.flatten()
        all_rssi = all_rssi[~np.isnan(all_rssi)]  # Remove NaN values
        
        if len(all_rssi) == 0:
            return np.zeros(len(self.percentiles) + 5)
        
        features = [
            np.min(all_rssi),
            np.max(all_rssi),
            np.mean(all_rssi),
            np.std(all_rssi)
        ]
        
        # Percentiles
        for p in self.percentiles:
            features.append(np.percentile(all_rssi, p))
        
        # Number of visible access points
        features.append(wifi_data.shape[1])
        
        return np.array(features)
    
    def extract_window_features(
        self,
        imu_data: np.ndarray,
        mag_data: np.ndarray,
        wifi_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract all features from a single window of heterogeneous sensor data.
        
        Args:
            imu_data: Array of shape (window_size, 6)
            mag_data: Array of shape (window_size, 3)
            wifi_data: Optional array of shape (window_size, n_aps)
        
        Returns:
            Concatenated feature vector
        """
        features = []
        
        # IMU features
        imu_features = self.extract_imu_features(imu_data)
        features.append(imu_features)
        
        # Magnetic features
        mag_features = self.extract_magnetic_features(mag_data)
        features.append(mag_features)
        
        # Wi-Fi features (optional)
        if wifi_data is not None and len(wifi_data) > 0:
            wifi_features = self.extract_wifi_features(wifi_data)
            features.append(wifi_features)
        
        return np.concatenate(features)
    
    def extract_sequence_features(
        self,
        imu_sequence: np.ndarray,
        mag_sequence: np.ndarray,
        wifi_sequence: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from entire sequences using sliding windows.
        
        Args:
            imu_sequence: Array of shape (n_samples, 6)
            mag_sequence: Array of shape (n_samples, 3)
            wifi_sequence: Optional array of shape (n_samples, n_aps)
        
        Returns:
            Tuple of (features, timestamps)
            - features: Array of shape (n_windows, n_features)
            - timestamps: Array of window center timestamps
        """
        n_samples = len(imu_sequence)
        
        if len(mag_sequence) != n_samples:
            raise ValueError("IMU and magnetic sequences must have same length")
        
        if wifi_sequence is not None and len(wifi_sequence) != n_samples:
            raise ValueError("Wi-Fi sequence must have same length as IMU/magnetic")
        
        features_list = []
        timestamps = []
        
        # Sliding window extraction
        for start_idx in range(0, n_samples - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            
            # Extract window
            imu_window = imu_sequence[start_idx:end_idx]
            mag_window = mag_sequence[start_idx:end_idx]
            wifi_window = wifi_sequence[start_idx:end_idx] if wifi_sequence is not None else None
            
            # Extract features
            window_features = self.extract_window_features(
                imu_window, mag_window, wifi_window
            )
            features_list.append(window_features)
            
            # Record center timestamp
            timestamps.append(start_idx + self.window_size // 2)
        
        return np.array(features_list), np.array(timestamps)
    
    def get_feature_dimension(self, include_wifi: bool = True) -> int:
        """
        Get the total number of features.
        
        Args:
            include_wifi: Whether Wi-Fi features are included
        
        Returns:
            Total feature dimension
        """
        n_stats = 4 + len(self.percentiles)  # min, max, mean, std + percentiles
        
        # IMU: 6 axes * n_stats + 4 magnitude features
        imu_dim = 6 * n_stats + 4
        
        # Magnetic: 3 axes * n_stats + 2 magnitude features
        mag_dim = 3 * n_stats + 2
        
        # Wi-Fi: n_stats + 1 (AP count)
        wifi_dim = n_stats + 1 if include_wifi else 0
        
        return imu_dim + mag_dim + wifi_dim


def create_sample_data(n_samples: int = 200) -> Dict[str, np.ndarray]:
    """
    Create sample sensor data for testing.
    
    Args:
        n_samples: Number of samples to generate
    
    Returns:
        Dictionary with 'imu', 'magnetic', and 'wifi' data
    """
    # Simulate IMU data (accelerometer + gyroscope)
    imu_data = np.random.randn(n_samples, 6) * 0.5
    imu_data[:, :3] += [0, 0, 9.81]  # Add gravity to accelerometer
    
    # Simulate magnetic field data
    mag_data = np.random.randn(n_samples, 3) * 5 + [20, 30, -40]
    
    # Simulate Wi-Fi RSSI data (5 access points)
    wifi_data = np.random.randn(n_samples, 5) * 10 - 70
    
    return {
        'imu': imu_data,
        'magnetic': mag_data,
        'wifi': wifi_data
    }


if __name__ == "__main__":
    # Example usage
    print("=== Sensor Feature Extraction Demo ===\n")
    
    # Create feature extractor
    extractor = SensorFeatureExtractor(window_size=50, stride=25)
    
    print(f"Window size: {extractor.window_size}")
    print(f"Stride: {extractor.stride}")
    print(f"Feature dimension (with Wi-Fi): {extractor.get_feature_dimension(True)}")
    print(f"Feature dimension (without Wi-Fi): {extractor.get_feature_dimension(False)}\n")
    
    # Generate sample data
    sample_data = create_sample_data(n_samples=200)
    
    print(f"Sample data shapes:")
    print(f"  IMU: {sample_data['imu'].shape}")
    print(f"  Magnetic: {sample_data['magnetic'].shape}")
    print(f"  Wi-Fi: {sample_data['wifi'].shape}\n")
    
    # Extract features from sequences
    features, timestamps = extractor.extract_sequence_features(
        sample_data['imu'],
        sample_data['magnetic'],
        sample_data['wifi']
    )
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Number of windows: {len(timestamps)}")
    print(f"Timestamps: {timestamps[:5]}...\n")
    
    # Show sample features
    print("Sample feature vector (first window):")
    print(f"  Min: {features[0].min():.3f}")
    print(f"  Max: {features[0].max():.3f}")
    print(f"  Mean: {features[0].mean():.3f}")
    print(f"  Std: {features[0].std():.3f}")
