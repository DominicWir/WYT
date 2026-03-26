#!/usr/bin/env python3
"""
Convert PyTorch model to TensorFlow Lite for Android deployment.
This avoids the 16KB page alignment issues with PyTorch Mobile.
"""

import torch
import tensorflow as tf
import numpy as np
from inference import load_trained_model
import onnx
from onnx_tf.backend import prepare

def convert_pytorch_to_tflite(
    pytorch_model_path: str,
    output_tflite_path: str,
    sequence_length: int = 50,
    feature_dim: int = 69
):
    """
    Convert PyTorch model to TensorFlow Lite.
    
    Args:
        pytorch_model_path: Path to best_model.pt
        output_tflite_path: Output path for .tflite file
        sequence_length: Input sequence length (50)
        feature_dim: Number of features (69)
    """
    
    print("Step 1: Loading PyTorch model...")
    model = load_trained_model(pytorch_model_path)
    model.eval()
    model = model.cpu()
    
    print("Step 2: Exporting to ONNX...")
    dummy_input = torch.randn(1, sequence_length, feature_dim)
    onnx_path = pytorch_model_path.replace('.pt', '.onnx')
    
    # Export to ONNX with simplified output
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            # Only return activity predictions (not position or uncertainty)
            _, activity, _ = self.model(x, return_uncertainty=False)
            return activity
    
    wrapped_model = ModelWrapper(model)
    
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['activity'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'activity': {0: 'batch_size'}
        }
    )
    print(f"✓ ONNX model saved to: {onnx_path}")
    
    print("Step 3: Converting ONNX to TensorFlow...")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_model_path = pytorch_model_path.replace('.pt', '_tf')
    tf_rep.export_graph(tf_model_path)
    print(f"✓ TensorFlow model saved to: {tf_model_path}")
    
    print("Step 4: Converting TensorFlow to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    # Optimize for mobile
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open(output_tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ TFLite model saved to: {output_tflite_path}")
    print(f"✓ Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    # Verify the model
    print("\nStep 5: Verifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=output_tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Test inference
    test_input = np.random.randn(1, sequence_length, feature_dim).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Test output shape: {output.shape}")
    print("✓ Model verification successful!")
    
    return output_tflite_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python convert_to_tflite.py <path_to_best_model.pt>")
        print("\nExample:")
        print("  python convert_to_tflite.py checkpoints_activity/best_model.pt")
        sys.exit(1)
    
    pytorch_model = sys.argv[1]
    tflite_output = pytorch_model.replace('.pt', '.tflite')
    
    try:
        convert_pytorch_to_tflite(pytorch_model, tflite_output)
        print(f"\n✅ Conversion complete!")
        print(f"\nNext steps:")
        print(f"1. Copy {tflite_output} to Android assets/")
        print(f"2. Update Android code to use TFLite instead of PyTorch")
        print(f"3. Run on Pixel 8 without any issues!")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print("\nTrying alternative method...")
        print("You may need to install: pip install onnx tf2onnx onnx-tf tensorflow")
        sys.exit(1)
