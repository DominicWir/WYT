# Dual Model Training Guide

## Overview

We're training **two specialized models** instead of one multi-task model:

1. **Activity Recognition** - Using Opportunity dataset (real IMU data)
2. **Indoor Positioning** - Using UJIIndoorLoc dataset (real Wi-Fi data)

This approach gives better results because each model focuses on its specific task.

## Model 1: Activity Recognition

### Dataset
- **Source**: Opportunity Activity Recognition
- **Sensors**: Real IMU (accelerometer + gyroscope) + magnetometer
- **Samples**: ~500,000+ from 4 subjects, 5 sessions each
- **Activities**: Stationary, Walking, Running, Stairs Up, Stairs Down

### Training
```bash
cd /Users/dominicwirasinha/AndroidStudioProjects/WYTV2
source .venv/bin/activate
cd ml
python train_activity.py
```

### Output
- `checkpoints_activity/best_model.pt`
- `activity_recognition.onnx` (for Android)
- `activity_normalizer.json`

### Expected Performance
- **Activity accuracy**: 85-95%
- **Training time**: ~30-60 minutes (30 epochs)

---

## Model 2: Indoor Positioning

### Dataset
- **Source**: UJIIndoorLoc
- **Sensors**: 520 Wi-Fi access points (RSSI values)
- **Samples**: 19,937 training samples
- **Coverage**: 3 buildings, multiple floors
- **Positions**: Real (longitude, latitude, floor)

### Training
```bash
cd /Users/dominicwirasinha/AndroidStudioProjects/WYTV2
source .venv/bin/activate
cd ml
python train_positioning.py
```

### Output
- `checkpoints_positioning/best_model.pt`
- `indoor_positioning.onnx` (for Android)
- `positioning_normalizer.json`

### Expected Performance
- **Position accuracy**: 5-10 meters
- **Floor accuracy**: 95%+
- **Training time**: ~20-40 minutes (30 epochs)

---

## Training Both Models

Run them sequentially:

```bash
# Terminal 1: Activity recognition
python train_activity.py

# After it completes, Terminal 2: Positioning
python train_positioning.py
```

Or run in parallel (if you have enough CPU):
```bash
# Terminal 1
python train_activity.py &

# Terminal 2  
python train_positioning.py &
```

---

## Android Integration

You'll have **two ONNX models** to integrate:

### 1. Activity Recognition Model
```kotlin
// Load activity model
val activitySession = OrtEnvironment.getEnvironment().createSession(
    "activity_recognition.onnx",
    OrtSession.SessionOptions()
)

// Run inference on IMU data
val activityResult = activitySession.run(imuFeatures)
val predictedActivity = activityResult.get(1) // 0-4
```

### 2. Positioning Model
```kotlin
// Load positioning model
val positionSession = OrtEnvironment.getEnvironment().createSession(
    "indoor_positioning.onnx",
    OrtSession.SessionOptions()
)

// Run inference on Wi-Fi data
val positionResult = positionSession.run(wifiFeatures)
val predictedPosition = positionResult.get(0) // (x, y, floor)
```

### Combined Usage
```kotlin
// Get both predictions
val activity = getActivityPrediction(imuData)
val position = getPositionPrediction(wifiData)

// Use together
updateUI(position, activity)
```

---

## Configuration

Both scripts use `config.yaml`. Key settings:

```yaml
training:
  num_epochs: 30  # Adjust for better accuracy
  batch_size: 32  # Reduce if out of memory
  sequence_length: 50

model:
  hidden_size: 256
  num_layers: 3
```

---

## Monitoring Training

Watch progress:
```bash
# Activity training
tail -f checkpoints_activity/training_history.json

# Positioning training
tail -f checkpoints_positioning/training_history.json
```

---

## Next Steps

1. **Train activity model**: `python train_activity.py`
2. **Train positioning model**: `python train_positioning.py`
3. **Test models**: Use `inference.py` to verify
4. **Deploy to Android**: Copy both ONNX files
5. **Collect your own data**: For even better results!

---

## Advantages of Dual Models

✅ **Better accuracy** - Each model specializes
✅ **Faster training** - Smaller, focused models
✅ **Easier debugging** - Test each independently
✅ **Flexible deployment** - Use one or both
✅ **Real data** - Both use actual sensor measurements
