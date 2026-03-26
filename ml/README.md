# xLSTM Sensor Fusion for Indoor Positioning

This module implements an **Extended LSTM (xLSTM)** architecture in PyTorch for processing heterogeneous sensor data from IMU, Magnetic field sensors, and Wi-Fi signals. The system extracts statistical features and learns temporal dependencies for indoor positioning and activity recognition.

## 🚀 Quick Start

### 1. Installation

```bash
cd /Users/dominicwirasinha/AndroidStudioProjects/WYTV2/ml

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Example

```bash
python example_usage.py
```

This will demonstrate the complete workflow with sample data.

## 📁 Project Structure

```
ml/
├── feature_extraction.py    # Statistical feature extraction
├── xlstm_model.py           # xLSTM architecture implementation
├── data_preprocessing.py    # Data normalization and preprocessing
├── train.py                 # Training script
├── inference.py             # Inference and model export
├── config.yaml              # Configuration file
├── example_usage.py         # Complete workflow examples
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Components

### Feature Extraction

Extract statistical features (min, max, mean, std, percentiles) from sensor windows:

```python
from feature_extraction import SensorFeatureExtractor

extractor = SensorFeatureExtractor(
    window_size=50,    # ~1 second at 50Hz
    stride=25,         # 50% overlap
    percentiles=[25.0, 50.0, 75.0]
)

features, timestamps = extractor.extract_sequence_features(
    imu_data,      # Shape: (n_samples, 6)
    mag_data,      # Shape: (n_samples, 3)
    wifi_data      # Shape: (n_samples, n_aps)
)
```

**Supported Sensors:**
- **IMU**: 6-axis (accelerometer + gyroscope)
- **Magnetic**: 3-axis magnetometer
- **Wi-Fi**: RSSI from multiple access points

### xLSTM Model

The model uses **mLSTM** (matrix memory LSTM) for better temporal modeling:

```python
from xlstm_model import create_model

model = create_model(
    feature_dim=70,
    config={
        'hidden_size': 256,
        'num_layers': 3,
        'num_attention_heads': 8
    }
)
```

**Key Features:**
- Matrix memory for multivariate data
- Exponential gating for better gradients
- Multi-head attention for sensor fusion
- Multi-task outputs: position + activity

### Training

```python
from train import Trainer

trainer = Trainer(model, train_loader, val_loader, config)
trainer.train(num_epochs=100)
```

**Features:**
- Multi-task learning (position + activity)
- Automatic checkpointing
- Learning rate scheduling
- Gradient clipping

### Inference

**Streaming inference** for real-time predictions:

```python
from inference import StreamingInference

streaming = StreamingInference(model, feature_extractor, normalizer)

# Add samples one at a time
position, activity, uncertainty = streaming.predict(
    imu_sample, mag_sample, wifi_sample
)
```

**Model export** for deployment:

```python
from inference import ModelExporter

exporter = ModelExporter(model)
exporter.export_to_onnx("model.onnx")
```

## 📊 Step-by-Step Instructions

### Step 1: Prepare Your Data

Collect sensor data from your Android app and save as NumPy arrays:

```python
import numpy as np

# Your sensor data
imu_data = np.array([...])      # Shape: (n_samples, 6)
mag_data = np.array([...])      # Shape: (n_samples, 3)
wifi_data = np.array([...])     # Shape: (n_samples, n_aps)

# Ground truth labels
positions = np.array([...])     # Shape: (n_samples, 3) - (x, y, floor)
activities = np.array([...])    # Shape: (n_samples,) - activity class IDs
```

### Step 2: Extract Features

```python
from feature_extraction import SensorFeatureExtractor

extractor = SensorFeatureExtractor(window_size=50, stride=25)
features, timestamps = extractor.extract_sequence_features(
    imu_data, mag_data, wifi_data
)

# Align labels with feature timestamps
positions_aligned = positions[timestamps]
activities_aligned = activities[timestamps]
```

### Step 3: Preprocess Data

```python
from data_preprocessing import (
    SensorDataNormalizer,
    create_train_val_split,
    create_dataloaders
)

# Normalize features
normalizer = SensorDataNormalizer()
normalizer.fit(imu_data, mag_data, wifi_data)
normalizer.save('normalizer_params.json')

# Split data
train_data, val_data = create_train_val_split(
    features, positions_aligned, activities_aligned, val_ratio=0.2
)

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    train_data, val_data, batch_size=32, sequence_length=20
)
```

### Step 4: Configure and Train

Edit `config.yaml` to adjust hyperparameters, then train:

```python
from xlstm_model import create_model
from train import Trainer
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = create_model(
    feature_dim=features.shape[1],
    config=config['model']
)

# Train
trainer = Trainer(model, train_loader, val_loader, config['training'])
trainer.train(num_epochs=100)
```

### Step 5: Evaluate and Export

```python
from inference import load_trained_model, ModelExporter

# Load best model
model = load_trained_model('checkpoints/best_model.pt')

# Export to ONNX for Android
exporter = ModelExporter(model)
exporter.export_to_onnx('sensor_fusion_xlstm.onnx')

# Optional: Export to TFLite
exporter.export_to_tflite('sensor_fusion_xlstm.onnx', 'model.tflite')
```

### Step 6: Integrate with Android

1. **Copy ONNX model** to your Android project: `app/src/main/assets/`

2. **Add ONNX Runtime dependency** to `app/build.gradle.kts`:
   ```kotlin
   dependencies {
       implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.0")
   }
   ```

3. **Load and run inference** in your Android app:
   ```java
   // Load model
   OrtEnvironment env = OrtEnvironment.getEnvironment();
   OrtSession session = env.createSession("model.onnx");
   
   // Prepare input (features from your sensors)
   float[][][] input = extractFeatures(sensorData);
   
   // Run inference
   OnnxTensor inputTensor = OnnxTensor.createTensor(env, input);
   OrtSession.Result results = session.run(Collections.singletonMap("sensor_features", inputTensor));
   
   // Get outputs
   float[][] position = results.get("position").getValue();
   float[][] activityLogits = results.get("activity_logits").getValue();
   ```

## 🎯 Configuration

Key parameters in `config.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `window_size` | Samples per feature window | 50 |
| `hidden_size` | LSTM hidden dimension | 256 |
| `num_layers` | Number of LSTM layers | 3 |
| `batch_size` | Training batch size | 32 |
| `learning_rate` | Initial learning rate | 0.001 |
| `position_weight` | Position loss weight | 1.0 |
| `activity_weight` | Activity loss weight | 0.5 |

## 📈 Expected Performance

With proper training data:
- **Position accuracy**: ~2-3 meters (depending on environment)
- **Activity recognition**: >90% accuracy
- **Inference time**: <10ms per prediction (CPU)

## 🔍 Troubleshooting

**Issue**: Out of memory during training
- **Solution**: Reduce `batch_size` or `hidden_size` in config.yaml

**Issue**: Poor convergence
- **Solution**: Increase `num_epochs`, adjust `learning_rate`, or collect more training data

**Issue**: NaN losses
- **Solution**: Enable `grad_clip` in config.yaml, reduce learning rate

## 📚 References

- Beck et al. "xLSTM: Extended Long Short-Term Memory" (2024)
- Original LSTM: Hochreiter & Schmidhuber (1997)

## 🤝 Integration with Your App

This ML module is designed to work with your **WYT V2** Android app:

1. Collect sensor data using your existing `StepDetectionService`
2. Export data periodically for training
3. Train the xLSTM model offline
4. Deploy the ONNX model back to the app
5. Use predictions to enhance your PDR system

## 📝 License

Part of the WYT V2 project.
