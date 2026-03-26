"""
Training Script for Activity Recognition with FORCED GPU (MPS) Usage

This script explicitly forces GPU (MPS) usage for training.
"""

import numpy as np
import yaml
from pathlib import Path
import torch

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


def main():
    """Train activity recognition model on Opportunity dataset with GPU."""
    
    print("=" * 60)
    print("ACTIVITY RECOGNITION TRAINING (Opportunity Dataset)")
    print("GPU (MPS) FORCED MODE")
    print("=" * 60 + "\n")
    
    # Force MPS device
    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available!")
        print("Falling back to CPU...")
        device = 'cpu'
    else:
        device = 'mps'
        print(f"✓ MPS is available and will be used!")
        print(f"✓ MPS built: {torch.backends.mps.is_built()}\n")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded from config.yaml\n")
    
    # ========================================
    # LOAD OPPORTUNITY DATASET
    # ========================================
    
    print("Loading Opportunity dataset...")
    
    from load_zip_datasets import load_dataset_from_zip
    
    data_path = "/Users/dominicwirasinha/Documents/IIT/Year 5/FYP Dev/opportunity+activity+recognition.zip"
    
    # Use all subjects and sessions for better activity recognition
    dataset = load_dataset_from_zip(
        'opportunity', 
        data_path,
        subjects=['S1', 'S2', 'S3', 'S4'],  # All subjects
        sessions=['ADL1', 'ADL2', 'ADL3', 'ADL4', 'ADL5']  # All ADL sessions
    )
    
    imu_data = dataset['imu']
    mag_data = dataset['magnetic']
    activities = dataset['activities']
    
    print(f"Dataset loaded:")
    print(f"  IMU shape: {imu_data.shape}")
    print(f"  Magnetic shape: {mag_data.shape}")
    print(f"  Activities shape: {activities.shape}")
    print(f"  Total samples: {len(imu_data)}\n")
    
    # ========================================
    # PREPROCESSING
    # ========================================
    
    print("Preprocessing data...")
    
    imu_clean, mag_clean, _ = handle_missing_data(
        imu_data, mag_data, None,
        method=config['preprocessing']['missing_data_method']
    )
    
    if config['preprocessing'].get('outlier_threshold', 0) > 0:
        imu_clean = remove_outliers(imu_clean, config['preprocessing']['outlier_threshold'])
        mag_clean = remove_outliers(mag_clean, config['preprocessing']['outlier_threshold'])
    
    normalizer = SensorDataNormalizer()
    normalizer.fit(imu_clean, mag_clean, None)
    imu_norm, mag_norm, _ = normalizer.transform(imu_clean, mag_clean, None)
    
    normalizer.save('activity_normalizer.json')
    print("  ✓ Normalizer saved to activity_normalizer.json")
    
    # ========================================
    # FEATURE EXTRACTION
    # ========================================
    
    print("\nExtracting features...")
    
    extractor = SensorFeatureExtractor(
        window_size=config['feature_extraction']['window_size'],
        stride=config['feature_extraction']['stride'],
        percentiles=config['feature_extraction']['percentiles']
    )
    
    features, timestamps = extractor.extract_sequence_features(
        imu_norm, mag_norm, None  # No Wi-Fi data
    )
    
    activities_aligned = activities[timestamps]
    
    # Create dummy positions (not used for activity recognition)
    positions_dummy = np.zeros((len(features), 3))
    
    print(f"  Features extracted: {features.shape}")
    print(f"  Feature dimension: {features.shape[1]}\n")
    
    # ========================================
    # TRAIN/VAL SPLIT
    # ========================================
    
    print("Splitting data...")
    
    train_data, val_data = create_train_val_split(
        features,
        positions_dummy,
        activities_aligned,
        val_ratio=config['training']['val_ratio'],
        random_seed=config['training']['random_seed']
    )
    
    print(f"  Training samples: {len(train_data['features'])}")
    print(f"  Validation samples: {len(val_data['features'])}\n")
    
    # ========================================
    # CREATE DATALOADERS
    # ========================================
    
    train_loader, val_loader = create_dataloaders(
        train_data, val_data,
        batch_size=config['training']['batch_size'],
        sequence_length=config['training']['sequence_length'],
        num_workers=0
    )
    
    print(f"DataLoaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}\n")
    
    # ========================================
    # CREATE MODEL
    # ========================================
    
    print("Creating model...")
    
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
    # TRAIN MODEL (ACTIVITY RECOGNITION ONLY)
    # ========================================
    
    print("Starting training...\n")
    print("=" * 60)
    
    # Configure for activity recognition only
    train_config = {
        'feature_dim': features.shape[1],
        **config['training'],
        'position_weight': 0.0,  # Disable position loss
        'activity_weight': 1.0,  # Focus on activity
        'checkpoint_dir': 'checkpoints_activity'
    }
    
    # FORCE GPU DEVICE
    trainer = Trainer(model, train_loader, val_loader, train_config, device=device)
    
    print(f"\n✓✓✓ TRAINING ON DEVICE: {trainer.device} ✓✓✓\n")
    
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    print("=" * 60)
    print("\n✓ Activity recognition training completed!")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"  Model saved to: checkpoints_activity/best_model.pt")
    print(f"  Training history: checkpoints_activity/training_history.json\n")
    
    # ========================================
    # EXPORT MODEL
    # ========================================
    
    print("Exporting model to ONNX...")
    
    from inference import ModelExporter, load_trained_model
    
    best_model = load_trained_model("checkpoints_activity/best_model.pt")
    
    exporter = ModelExporter(best_model)
    exporter.export_to_onnx(
        'activity_recognition.onnx',
        input_shape=(1, config['training']['sequence_length'], features.shape[1])
    )
    
    print("\n✓ Activity recognition model ready!")
    print("\nNext: Train positioning model with train_positioning.py")


if __name__ == "__main__":
    main()
