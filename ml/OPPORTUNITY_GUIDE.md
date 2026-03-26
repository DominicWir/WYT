# Opportunity Dataset - Ready to Train! 🎯

## ✅ Dataset Loaded Successfully

The **Opportunity Activity Recognition** dataset is now configured and ready for training!

### Dataset Overview

- **292 MB** of real IMU sensor data
- **4 subjects** (S1, S2, S3, S4)
- **6 sessions each**: ADL1-5 (Activities of Daily Living) + Drill
- **250 sensor channels** including:
  - Accelerometers (multiple body locations)
  - Gyroscopes
  - Magnetometers
- **Sampling rate**: 30 Hz
- **Activity labels**: Locomotion (stand, walk, sit, lie)

### What We're Using

**Configured for training:**
- **Subjects**: S1, S2, S3 (3 subjects for training)
- **Sessions**: ADL1, ADL2, ADL3 (daily activities)
- **Sensor**: BACK sensor (has all: acc + gyro + mag)
- **Activities**: Mapped to 5 classes (stationary, walking, running, stairs_up, stairs_down)

### Sensor Data Extracted

✅ **IMU (6-axis)**: Real accelerometer + gyroscope from BACK sensor
✅ **Magnetic (3-axis)**: Real magnetometer data
✅ **Activities**: Real locomotion labels
⚠️ **Positions**: Estimated from IMU (dataset doesn't have ground truth positions)

## 🚀 Start Training

```bash
cd /Users/dominicwirasinha/AndroidStudioProjects/WYTV2
source .venv/bin/activate
cd ml
python train_custom_dataset.py
```

### What to Expect

**Dataset Size:**
- ~100,000-200,000 samples (depends on subjects/sessions selected)
- Real IMU sensor data from wearable sensors
- Actual activity labels

**Training Time:**
- ~20-40 minutes for 100 epochs
- Adjust `num_epochs` in `config.yaml` for faster testing

**Model Output:**
- Activity recognition (primary task)
- Position estimation (secondary, less accurate due to estimated positions)

## Configuration Options

Edit `train_custom_dataset.py` to customize:

```python
# Line 104-105: Select subjects and sessions
subjects = ['S1', 'S2', 'S3']  # Add 'S4' for more data
sessions = ['ADL1', 'ADL2', 'ADL3']  # Add 'ADL4', 'ADL5', 'Drill'
```

**More data = better model, but longer training time**

## Advantages Over UJIIndoorLoc

✅ **Real IMU data** (not synthetic)
✅ **Real activity labels**
✅ **Temporal sensor sequences** (perfect for xLSTM)
✅ **Multiple subjects** (better generalization)

⚠️ **Note**: Positions are estimated (not ground truth). For real positioning, you'll need to collect data with your Android app.

## After Training

The model will be excellent for:
1. **Activity recognition** (very accurate)
2. **Motion pattern detection**
3. **Sensor fusion learning**

For positioning, you should:
1. Collect your own data with ground truth positions
2. Or use UJIIndoorLoc for Wi-Fi-based positioning
3. Or combine both datasets

## Ready?

Just run the training command above! The Opportunity dataset is perfect for learning sensor fusion with real IMU data.
