"""
Training Script for Indoor Positioning (UJIIndoorLoc Dataset)

This script trains an xLSTM model specifically for indoor positioning
using Wi-Fi fingerprinting data from the UJIIndoorLoc dataset.
"""

import numpy as np
import yaml
from pathlib import Path

from feature_extraction import SensorFeatureExtractor
from data_preprocessing import (
    SensorDataNormalizer,
    create_train_val_split,
    create_dataloaders,
    handle_missing_data
)
from xlstm_model import create_model
from train import Trainer


def main():
    """Train positioning model on UJIIndoorLoc dataset."""
    
    print("=" * 60)
    print("INDOOR POSITIONING TRAINING (UJIIndoorLoc Dataset)")
    print("=" * 60 + "\n")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded from config.yaml\n")
    
    # ========================================
    # LOAD UJIINDOORLOC DATASET
    # ========================================
    
    print("Loading UJIIndoorLoc dataset...")
    
    from load_zip_datasets import load_dataset_from_zip
    
    data_path = "/Users/dominicwirasinha/Documents/IIT/Year 5/FYP Dev/ujiindoorloc.zip"
    
    # Load training set
    dataset = load_dataset_from_zip('ujiindoorloc', data_path, use_validation=False)
    
    imu_data = dataset['imu']  # Synthetic (for compatibility)
    mag_data = dataset['magnetic']  # Synthetic
    wifi_data = dataset['wifi']  # Real Wi-Fi RSSI
    positions = dataset['positions']  # Real positions
    activities = dataset['activities']  # Dummy
    
    print(f"Dataset loaded:")
    print(f"  Wi-Fi shape: {wifi_data.shape}")
    print(f"  Positions shape: {positions.shape}")
    print(f"  Total samples: {len(wifi_data)}\n")
    
    # ========================================
    # PREPROCESSING
    # ========================================
    
    print("Preprocessing data...")
    
    imu_clean, mag_clean, wifi_clean = handle_missing_data(
        imu_data, mag_data, wifi_data,
        method=config['preprocessing']['missing_data_method']
    )
    
    normalizer = SensorDataNormalizer()
    normalizer.fit(imu_clean, mag_clean, wifi_clean)
    imu_norm, mag_norm, wifi_norm = normalizer.transform(imu_clean, mag_clean, wifi_clean)
    
    normalizer.save('positioning_normalizer.json')
    print("  ✓ Normalizer saved to positioning_normalizer.json")
    
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
        imu_norm, mag_norm, wifi_norm
    )
    
    positions_aligned = positions[timestamps]
    activities_aligned = activities[timestamps]
    
    print(f"  Features extracted: {features.shape}")
    print(f"  Feature dimension: {features.shape[1]}\n")
    
    # ========================================
    # TRAIN/VAL SPLIT
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
    # TRAIN MODEL (POSITIONING ONLY)
    # ========================================
    
    print("Starting training...\n")
    print("=" * 60)
    
    # Configure for positioning only
    train_config = {
        'feature_dim': features.shape[1],
        **config['training'],
        'position_weight': 1.0,  # Focus on position
        'activity_weight': 0.0,  # Disable activity loss
        'checkpoint_dir': 'checkpoints_positioning'
    }
    
    trainer = Trainer(model, train_loader, val_loader, train_config)
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    print("=" * 60)
    print("\n✓ Indoor positioning training completed!")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"  Model saved to: checkpoints_positioning/best_model.pt")
    print(f"  Training history: checkpoints_positioning/training_history.json\n")
    
    # ========================================
    # EXPORT MODEL
    # ========================================
    
    print("Exporting model to ONNX...")
    
    from inference import ModelExporter, load_trained_model
    
    best_model = load_trained_model("checkpoints_positioning/best_model.pt")
    
    exporter = ModelExporter(best_model)
    exporter.export_to_onnx(
        'indoor_positioning.onnx',
        input_shape=(1, config['training']['sequence_length'], features.shape[1])
    )
    
    print("\n✓ Indoor positioning model ready!")
    print("\nYou now have two specialized models:")
    print("  1. activity_recognition.onnx - For activity detection")
    print("  2. indoor_positioning.onnx - For Wi-Fi positioning")


if __name__ == "__main__":
    main()
