"""
Training Script for Your Custom Dataset

This script shows how to train the xLSTM model with your own sensor data.
Adapt this to your specific dataset format.
"""

import numpy as np
import yaml
from pathlib import Path

from feature_extraction import SensorFeatureExtractor
from data_preprocessing import (
    SensorDataNormalizer,
    create_train_val_split,
    create_dataloaders,
    handle_missing_data,
    remove_outliers
)
from xlstm_model import create_model
from train import Trainer


def load_your_dataset(data_path: str, dataset_type: str = 'ujiindoorloc'):
    """
    Load your custom dataset from ZIP file.
    
    Args:
        data_path: Path to your ZIP dataset file
        dataset_type: 'ujiindoorloc' or 'opportunity' or 'custom'
    
    Returns:
        Dictionary with sensor data and labels
    """
    from load_zip_datasets import load_dataset_from_zip
    
    # If it's a ZIP file, use the ZIP loader
    if data_path.endswith('.zip'):
        print(f"Loading {dataset_type} dataset from ZIP...")
        return load_dataset_from_zip(dataset_type, data_path)
    
    # Otherwise, load from individual files
    data_path = Path(data_path)
    
    # Example: Load from .npy files
    # Adjust these to match your actual file names and formats
    imu_data = np.load(data_path / 'imu_data.npy')
    mag_data = np.load(data_path / 'magnetic_data.npy')
    
    # Wi-Fi data is optional
    wifi_path = data_path / 'wifi_data.npy'
    wifi_data = np.load(wifi_path) if wifi_path.exists() else None
    
    # Ground truth labels
    positions = np.load(data_path / 'positions.npy')  # (x, y, floor)
    activities = np.load(data_path / 'activities.npy')  # Activity class IDs
    
    # Alternative: Load from CSV
    # import pandas as pd
    # df = pd.read_csv(data_path / 'sensor_data.csv')
    # imu_data = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
    # mag_data = df[['mag_x', 'mag_y', 'mag_z']].values
    # positions = df[['x', 'y', 'floor']].values
    # activities = df['activity'].values
    
    return {
        'imu': imu_data,
        'magnetic': mag_data,
        'wifi': wifi_data,
        'positions': positions,
        'activities': activities
    }


def main():
    """Main training script for your dataset."""
    
    print("=" * 60)
    print("xLSTM TRAINING WITH CUSTOM DATASET")
    print("=" * 60 + "\n")
    
    # ========================================
    # 1. CONFIGURATION
    # ========================================
    
    # Load config from YAML
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded from config.yaml\n")
    
    # ========================================
    # 2. LOAD YOUR DATASET
    # ========================================
    
    print("Loading dataset...")
    
    # Dataset configuration
    data_path = "/Users/dominicwirasinha/Documents/IIT/Year 5/FYP Dev/opportunity+activity+recognition.zip"
    dataset_type = "opportunity"  # or "ujiindoorloc"
    
    # For Opportunity dataset, you can select specific subjects/sessions
    # None = load all
    subjects = ['S1', 'S2', 'S3']  # Use first 3 subjects for training
    sessions = ['ADL1', 'ADL2', 'ADL3']  # Use ADL sessions (not Drill)
    
    # Load your data
    from load_zip_datasets import load_dataset_from_zip
    dataset = load_dataset_from_zip(
        dataset_type, 
        data_path,
        subjects=subjects if dataset_type == 'opportunity' else None,
        sessions=sessions if dataset_type == 'opportunity' else None
    )
    
    imu_data = dataset['imu']
    mag_data = dataset['magnetic']
    wifi_data = dataset['wifi']
    positions = dataset['positions']
    activities = dataset['activities']
    
    print(f"Dataset loaded:")
    print(f"  IMU shape: {imu_data.shape}")
    print(f"  Magnetic shape: {mag_data.shape}")
    if wifi_data is not None:
        print(f"  Wi-Fi shape: {wifi_data.shape}")
    print(f"  Positions shape: {positions.shape}")
    print(f"  Activities shape: {activities.shape}")
    print(f"  Total samples: {len(imu_data)}\n")
    
    # ========================================
    # 3. DATA PREPROCESSING
    # ========================================
    
    print("Preprocessing data...")
    
    # Handle missing data
    imu_clean, mag_clean, wifi_clean = handle_missing_data(
        imu_data, mag_data, wifi_data,
        method=config['preprocessing']['missing_data_method']
    )
    
    # Remove outliers
    if config['preprocessing'].get('outlier_threshold', 0) > 0:
        imu_clean = remove_outliers(imu_clean, config['preprocessing']['outlier_threshold'])
        mag_clean = remove_outliers(mag_clean, config['preprocessing']['outlier_threshold'])
    
    # Normalize data
    normalizer = SensorDataNormalizer()
    normalizer.fit(imu_clean, mag_clean, wifi_clean)
    imu_norm, mag_norm, wifi_norm = normalizer.transform(imu_clean, mag_clean, wifi_clean)
    
    # Save normalizer for later use
    normalizer.save('normalizer_params.json')
    print("  ✓ Normalizer saved to normalizer_params.json")
    
    # ========================================
    # 4. FEATURE EXTRACTION
    # ========================================
    
    print("\nExtracting features...")
    
    extractor = SensorFeatureExtractor(
        window_size=config['feature_extraction']['window_size'],
        stride=config['feature_extraction']['stride'],
        percentiles=config['feature_extraction']['percentiles']
    )
    
    features, timestamps = extractor.extract_sequence_features(
        imu_norm, mag_norm, wifi_norm
    )
    
    # Align labels with feature timestamps
    positions_aligned = positions[timestamps]
    activities_aligned = activities[timestamps]
    
    print(f"  Features extracted: {features.shape}")
    print(f"  Feature dimension: {features.shape[1]}\n")
    
    # ========================================
    # 5. TRAIN/VAL SPLIT
    # ========================================
    
    print("Splitting data...")
    
    train_data, val_data = create_train_val_split(
        features,
        positions_aligned,
        activities_aligned,
        val_ratio=config['training']['val_ratio'],
        random_seed=config['training']['random_seed']
    )
    
    print(f"  Training samples: {len(train_data['features'])}")
    print(f"  Validation samples: {len(val_data['features'])}\n")
    
    # ========================================
    # 6. CREATE DATALOADERS
    # ========================================
    
    train_loader, val_loader = create_dataloaders(
        train_data, val_data,
        batch_size=config['training']['batch_size'],
        sequence_length=config['training']['sequence_length'],
        num_workers=0  # Set to 0 for debugging, increase for faster loading
    )
    
    print(f"DataLoaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}\n")
    
    # ========================================
    # 7. CREATE MODEL
    # ========================================
    
    print("Creating model...")
    
    # Prepare model config (exclude feature_dim as it's passed separately)
    model_config = {k: v for k, v in config['model'].items() if k != 'feature_dim'}
    
    model = create_model(
        feature_dim=features.shape[1],
        config=model_config
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Hidden size: {config['model']['hidden_size']}")
    print(f"  Number of layers: {config['model']['num_layers']}\n")
    
    # ========================================
    # 8. TRAIN MODEL
    # ========================================
    
    print("Starting training...\n")
    print("=" * 60)
    
    # Prepare training config
    train_config = {
        'feature_dim': features.shape[1],
        **config['training']
    }
    
    trainer = Trainer(model, train_loader, val_loader, train_config)
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    print("=" * 60)
    print("\n✓ Training completed!")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"  Model saved to: {train_config['checkpoint_dir']}/best_model.pt")
    print(f"  Training history: {train_config['checkpoint_dir']}/training_history.json\n")
    
    # ========================================
    # 9. EXPORT MODEL (OPTIONAL)
    # ========================================
    
    print("Exporting model to ONNX...")
    
    from inference import ModelExporter, load_trained_model
    
    # Load best model
    best_model = load_trained_model(
        f"{train_config['checkpoint_dir']}/best_model.pt"
    )
    
    # Export to ONNX
    exporter = ModelExporter(best_model)
    exporter.export_to_onnx(
        'sensor_fusion_xlstm.onnx',
        input_shape=(1, config['training']['sequence_length'], features.shape[1])
    )
    
    print("\n✓ All done!")
    print("\nNext steps:")
    print("1. Review training history in checkpoints/training_history.json")
    print("2. Test the model with inference.py")
    print("3. Integrate sensor_fusion_xlstm.onnx into your Android app")


if __name__ == "__main__":
    main()
