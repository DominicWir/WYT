"""
Inference and Model Export Utilities

This module provides utilities for real-time inference and exporting
the trained xLSTM model to ONNX/TFLite for Android deployment.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import onnx
import onnxruntime as ort
from pathlib import Path

from xlstm_model import SensorFusionXLSTM, create_model
from feature_extraction import SensorFeatureExtractor
from data_preprocessing import SensorDataNormalizer


class StreamingInference:
    """
    Real-time streaming inference for sensor data.
    """
    
    def __init__(
        self,
        model: SensorFusionXLSTM,
        feature_extractor: SensorFeatureExtractor,
        normalizer: SensorDataNormalizer,
        device: str = 'cpu'
    ):
        """
        Initialize streaming inference.
        
        Args:
            model: Trained xLSTM model
            feature_extractor: Feature extraction module
            normalizer: Data normalizer
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.feature_extractor = feature_extractor
        self.normalizer = normalizer
        self.device = device
        
        # Buffer for streaming data
        self.buffer_size = feature_extractor.window_size
        self.imu_buffer = []
        self.mag_buffer = []
        self.wifi_buffer = []
        
        # Feature buffer for sequence
        self.feature_buffer = []
        self.max_feature_buffer = 20  # Sequence length
    
    def reset(self):
        """Reset buffers."""
        self.imu_buffer = []
        self.mag_buffer = []
        self.wifi_buffer = []
        self.feature_buffer = []
    
    def add_sample(
        self,
        imu_sample: np.ndarray,
        mag_sample: np.ndarray,
        wifi_sample: Optional[np.ndarray] = None
    ):
        """
        Add a new sensor sample to the buffer.
        
        Args:
            imu_sample: IMU sample of shape (6,)
            mag_sample: Magnetic sample of shape (3,)
            wifi_sample: Optional Wi-Fi sample
        """
        self.imu_buffer.append(imu_sample)
        self.mag_buffer.append(mag_sample)
        
        if wifi_sample is not None:
            self.wifi_buffer.append(wifi_sample)
        
        # Keep buffer size limited
        if len(self.imu_buffer) > self.buffer_size:
            self.imu_buffer.pop(0)
            self.mag_buffer.pop(0)
            if wifi_sample is not None:
                self.wifi_buffer.pop(0)
    
    def can_extract_features(self) -> bool:
        """Check if buffer has enough samples for feature extraction."""
        return len(self.imu_buffer) >= self.buffer_size
    
    def extract_features(self) -> Optional[np.ndarray]:
        """
        Extract features from current buffer.
        
        Returns:
            Feature vector or None if buffer not ready
        """
        if not self.can_extract_features():
            return None
        
        # Convert buffers to arrays
        imu_window = np.array(self.imu_buffer[-self.buffer_size:])
        mag_window = np.array(self.mag_buffer[-self.buffer_size:])
        wifi_window = np.array(self.wifi_buffer[-self.buffer_size:]) if self.wifi_buffer else None
        
        # Normalize
        imu_norm, mag_norm, wifi_norm = self.normalizer.transform(
            imu_window, mag_window, wifi_window
        )
        
        # Extract features
        features = self.feature_extractor.extract_window_features(
            imu_norm, mag_norm, wifi_norm
        )
        
        return features
    
    def predict(
        self,
        imu_sample: np.ndarray,
        mag_sample: np.ndarray,
        wifi_sample: Optional[np.ndarray] = None
    ) -> Optional[Tuple[np.ndarray, int, Optional[np.ndarray]]]:
        """
        Make prediction from new sensor sample.
        
        Args:
            imu_sample: IMU sample
            mag_sample: Magnetic sample
            wifi_sample: Optional Wi-Fi sample
        
        Returns:
            Tuple of (position, activity_class, uncertainty) or None if not ready
        """
        # Add sample to buffer
        self.add_sample(imu_sample, mag_sample, wifi_sample)
        
        # Extract features if buffer is ready
        if not self.can_extract_features():
            return None
        
        features = self.extract_features()
        if features is None:
            return None
        
        # Add to feature buffer
        self.feature_buffer.append(features)
        if len(self.feature_buffer) > self.max_feature_buffer:
            self.feature_buffer.pop(0)
        
        # Need at least one feature for prediction
        if len(self.feature_buffer) == 0:
            return None
        
        # Prepare input tensor
        feature_sequence = np.array(self.feature_buffer)
        feature_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            position, activity_logits, uncertainty = self.model(
                feature_tensor,
                return_uncertainty=True
            )
        
        # Get latest predictions
        position_pred = position[0, -1, :].cpu().numpy()
        activity_pred = torch.argmax(activity_logits[0, -1, :]).item()
        uncertainty_pred = uncertainty[0, -1, :].cpu().numpy() if uncertainty is not None else None
        
        return position_pred, activity_pred, uncertainty_pred


class ModelExporter:
    """
    Export trained model to ONNX and TensorFlow Lite formats.
    """
    
    def __init__(self, model: SensorFusionXLSTM):
        self.model = model
        self.model.eval()
    
    def export_to_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, int, int] = (1, 20, 70),
        opset_version: int = 12
    ):
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Input shape (batch, seq_len, features)
            opset_version: ONNX opset version
        """
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['sensor_features'],
            output_names=['position', 'activity_logits', 'uncertainty'],
            dynamic_axes={
                'sensor_features': {0: 'batch_size', 1: 'sequence_length'},
                'position': {0: 'batch_size', 1: 'sequence_length'},
                'activity_logits': {0: 'batch_size', 1: 'sequence_length'},
                'uncertainty': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        print(f"✓ Model exported to ONNX: {output_path}")
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verified")
    
    def test_onnx_inference(
        self,
        onnx_path: str,
        test_input: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Test ONNX model inference.
        
        Args:
            onnx_path: Path to ONNX model
            test_input: Optional test input, otherwise random
        
        Returns:
            Dictionary of outputs
        """
        # Create session
        session = ort.InferenceSession(onnx_path)
        
        # Prepare input
        if test_input is None:
            input_shape = session.get_inputs()[0].shape
            # Replace dynamic dimensions with concrete values
            input_shape = [1 if isinstance(d, str) else d for d in input_shape]
            test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {'sensor_features': test_input})
        
        output_dict = {
            'position': outputs[0],
            'activity_logits': outputs[1],
            'uncertainty': outputs[2] if len(outputs) > 2 else None
        }
        
        print(f"✓ ONNX inference successful")
        print(f"  Position shape: {output_dict['position'].shape}")
        print(f"  Activity logits shape: {output_dict['activity_logits'].shape}")
        
        return output_dict
    
    def export_to_tflite(
        self,
        onnx_path: str,
        output_path: str
    ):
        """
        Convert ONNX model to TensorFlow Lite.
        
        Note: Requires onnx-tf and tensorflow packages.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TFLite model
        """
        try:
            import onnx
            from onnx_tf.backend import prepare
            import tensorflow as tf
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph('temp_tf_model')
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model('temp_tf_model')
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✓ Model exported to TFLite: {output_path}")
            
            # Clean up temp directory
            import shutil
            shutil.rmtree('temp_tf_model', ignore_errors=True)
            
        except ImportError as e:
            print(f"⚠ TFLite export requires onnx-tf and tensorflow packages")
            print(f"  Install with: pip install onnx-tf tensorflow")
            raise e


def load_trained_model(
    checkpoint_path: str,
    device: str = 'cpu'
) -> SensorFusionXLSTM:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint['config']
    
    # Create model
    model = create_model(
        feature_dim=config['feature_dim'],
        config={
            'hidden_size': config.get('hidden_size', 256),
            'num_layers': config.get('num_layers', 3)
        }
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    return model


if __name__ == "__main__":
    print("=== Inference and Export Demo ===\n")
    
    # Create a sample model
    feature_dim = 70
    model = create_model(feature_dim, config={'hidden_size': 128, 'num_layers': 2})
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Export to ONNX
    output_dir = Path("exports")
    output_dir.mkdir(exist_ok=True)
    
    onnx_path = output_dir / "sensor_fusion_xlstm.onnx"
    
    exporter = ModelExporter(model)
    exporter.export_to_onnx(str(onnx_path), input_shape=(1, 20, feature_dim))
    
    # Test ONNX inference
    print("\nTesting ONNX inference...")
    test_input = np.random.randn(1, 20, feature_dim).astype(np.float32)
    outputs = exporter.test_onnx_inference(str(onnx_path), test_input)
    
    print("\n✓ Export demo completed!")
    print(f"  ONNX model saved to: {onnx_path}")
