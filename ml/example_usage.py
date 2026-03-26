"""
Example Usage of xLSTM Sensor Fusion System

This script demonstrates the complete workflow:
1. Generate/load sensor data
2. Extract features
3. Train the model
4. Perform inference
5. Export the model
"""

import numpy as np
import torch
import yaml
from pathlib import Path

from feature_extraction import SensorFeatureExtractor, create_sample_data
from data_preprocessing import (
    SensorDataNormalizer,
    create_train_val_split,
    create_dataloaders,
    handle_missing_data
)
from xlstm_model import create_model
from train import Trainer
from inference import StreamingInference, ModelExporter, load_trained_model


def example_1_feature_extraction():
    """Example 1: Feature extraction from sensor data."""
    print("=" * 60)
    print("EXAMPLE 1: Feature Extraction")
    print("=" * 60 + "\n")
    
    # Create sample sensor data
    sample_data = create_sample_data(n_samples=200)
    
    # Initialize feature extractor
    extractor = SensorFeatureExtractor(
        window_size=50,
        stride=25,
        percentiles=[25.0, 50.0, 75.0]
    )
    
    # Extract features from sequences
    features, timestamps = extractor.extract_sequence_features(
        sample_data['imu'],
        sample_data['magnetic'],
        sample_data['wifi']
    )
    
    print(f"Input data:")
    print(f"  IMU shape: {sample_data['imu'].shape}")
    print(f"  Magnetic shape: {sample_data['magnetic'].shape}")
    print(f"  Wi-Fi shape: {sample_data['wifi'].shape}\n")
    
    print(f"Extracted features:")
    print(f"  Features shape: {features.shape}")
    print(f"  Number of windows: {len(timestamps)}")
    print(f"  Feature dimension: {extractor.get_feature_dimension(True)}\n")
    
    return features, sample_data


def example_2_data_preprocessing(features):
    """Example 2: Data preprocessing and normalization."""
    print("=" * 60)
    print("EXAMPLE 2: Data Preprocessing")
    print("=" * 60 + "\n")
    
    # Create dummy labels
    n_samples = len(features)
    positions = np.random.randn(n_samples, 3) * 10  # Random positions
    activities = np.random.randint(0, 5, n_samples)  # 5 activity classes
    
    # Split into train/val
    train_data, val_data = create_train_val_split(
        features, positions, activities, val_ratio=0.2
    )
    
    print(f"Data split:")
    print(f"  Training samples: {len(train_data['features'])}")
    print(f"  Validation samples: {len(val_data['features'])}\n")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_data, val_data,
        batch_size=16,
        sequence_length=20
    )
    
    print(f"DataLoaders:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}\n")
    
    return train_loader, val_loader


def example_3_model_training(train_loader, val_loader, feature_dim):
    """Example 3: Model training."""
    print("=" * 60)
    print("EXAMPLE 3: Model Training")
    print("=" * 60 + "\n")
    
    # Configuration
    config = {
        'feature_dim': feature_dim,
        'hidden_size': 128,
        'num_layers': 2,
        'batch_size': 16,
        'sequence_length': 20,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'num_epochs': 3,  # Short demo
        'position_weight': 1.0,
        'activity_weight': 0.5,
        'use_uncertainty': False,
        'grad_clip': 1.0,
        'save_every': 2,
        'checkpoint_dir': 'checkpoints'
    }
    
    # Create model
    model = create_model(
        feature_dim=config['feature_dim'],
        config={
            'hidden_size': config['hidden_size'],
            'num_layers': config['num_layers']
        }
    )
    
    print(f"Model created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Num layers: {config['num_layers']}\n")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train
    print("Starting training...\n")
    trainer.train(num_epochs=config['num_epochs'])
    
    return model, config


def example_4_streaming_inference(sample_data):
    """Example 4: Streaming inference."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Streaming Inference")
    print("=" * 60 + "\n")
    
    # Create components
    feature_extractor = SensorFeatureExtractor(window_size=50, stride=25)
    normalizer = SensorDataNormalizer()
    
    # Fit normalizer on sample data
    normalizer.fit(
        sample_data['imu'],
        sample_data['magnetic'],
        sample_data['wifi']
    )
    
    # Create a simple model for demo
    model = create_model(
        feature_dim=feature_extractor.get_feature_dimension(True),
        config={'hidden_size': 128, 'num_layers': 2}
    )
    
    # Initialize streaming inference
    streaming = StreamingInference(
        model, feature_extractor, normalizer, device='cpu'
    )
    
    print("Streaming inference initialized\n")
    print("Simulating real-time sensor stream...")
    
    # Simulate streaming data
    predictions = []
    for i in range(100):
        imu_sample = sample_data['imu'][i]
        mag_sample = sample_data['magnetic'][i]
        wifi_sample = sample_data['wifi'][i]
        
        result = streaming.predict(imu_sample, mag_sample, wifi_sample)
        
        if result is not None:
            position, activity, uncertainty = result
            predictions.append({
                'sample': i,
                'position': position,
                'activity': activity,
                'uncertainty': uncertainty
            })
    
    print(f"\nMade {len(predictions)} predictions from 100 samples")
    
    if predictions:
        last_pred = predictions[-1]
        print(f"\nLast prediction:")
        print(f"  Position: {last_pred['position']}")
        print(f"  Activity class: {last_pred['activity']}")
        print(f"  Uncertainty: {last_pred['uncertainty']}\n")


def example_5_model_export():
    """Example 5: Model export to ONNX."""
    print("=" * 60)
    print("EXAMPLE 5: Model Export")
    print("=" * 60 + "\n")
    
    # Create model
    feature_dim = 70
    model = create_model(
        feature_dim,
        config={'hidden_size': 128, 'num_layers': 2}
    )
    
    # Create export directory
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    
    # Export to ONNX
    onnx_path = export_dir / "sensor_fusion_xlstm.onnx"
    
    exporter = ModelExporter(model)
    exporter.export_to_onnx(
        str(onnx_path),
        input_shape=(1, 20, feature_dim)
    )
    
    # Test ONNX inference
    print("\nTesting ONNX inference...")
    test_input = np.random.randn(1, 20, feature_dim).astype(np.float32)
    outputs = exporter.test_onnx_inference(str(onnx_path), test_input)
    
    print(f"\n✓ Model exported and tested successfully!")
    print(f"  ONNX model: {onnx_path}\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("xLSTM SENSOR FUSION - COMPLETE WORKFLOW EXAMPLES")
    print("=" * 60 + "\n")
    
    # Example 1: Feature extraction
    features, sample_data = example_1_feature_extraction()
    
    # Example 2: Data preprocessing
    train_loader, val_loader = example_2_data_preprocessing(features)
    
    # Example 3: Model training
    feature_dim = features.shape[1]
    model, config = example_3_model_training(train_loader, val_loader, feature_dim)
    
    # Example 4: Streaming inference
    example_4_streaming_inference(sample_data)
    
    # Example 5: Model export
    example_5_model_export()
    
    print("=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")
    
    print("Next steps:")
    print("1. Replace sample data with your actual sensor recordings")
    print("2. Adjust hyperparameters in config.yaml")
    print("3. Train on real data: python train.py")
    print("4. Export model: python inference.py")
    print("5. Integrate ONNX model into your Android app\n")


if __name__ == "__main__":
    main()
