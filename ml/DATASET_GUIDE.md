# How to Train with Your Custom Dataset

## Quick Start

### Step 1: Prepare Your Data

Your dataset should contain:

1. **IMU Data** (accelerometer + gyroscope)
   - Shape: `(n_samples, 6)`
   - Columns: `[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`
   - Units: m/s² for accelerometer, rad/s for gyroscope

2. **Magnetic Field Data**
   - Shape: `(n_samples, 3)`
   - Columns: `[mag_x, mag_y, mag_z]`
   - Units: μT (microtesla)

3. **Wi-Fi Data** (optional)
   - Shape: `(n_samples, n_access_points)`
   - Values: RSSI in dBm (e.g., -70, -65, etc.)

4. **Position Labels**
   - Shape: `(n_samples, 3)`
   - Columns: `[x, y, floor]`
   - Units: meters for x,y; integer for floor

5. **Activity Labels**
   - Shape: `(n_samples,)`
   - Values: Integer class IDs (0-4)
   - Classes: 0=stationary, 1=walking, 2=running, 3=stairs_up, 4=stairs_down

### Step 2: Save Data in NumPy Format

```python
import numpy as np

# Save your data
np.save('dataset/imu_data.npy', imu_data)
np.save('dataset/magnetic_data.npy', mag_data)
np.save('dataset/wifi_data.npy', wifi_data)  # Optional
np.save('dataset/positions.npy', positions)
np.save('dataset/activities.npy', activities)
```

### Step 3: Update Training Script

Edit `train_custom_dataset.py`:

```python
# Line 88: Update this path
data_path = "path/to/your/dataset"  # Change to your actual path
```

### Step 4: Adjust Configuration (Optional)

Edit `config.yaml` to tune hyperparameters:

```yaml
# Key parameters to adjust
feature_extraction:
  window_size: 50      # Adjust based on your sampling rate
  stride: 25           # 50% overlap

training:
  batch_size: 32       # Reduce if out of memory
  num_epochs: 100      # Increase for better results
  learning_rate: 0.001 # Adjust if not converging
```

### Step 5: Run Training

```bash
cd /Users/dominicwirasinha/AndroidStudioProjects/WYTV2/ml
python train_custom_dataset.py
```

## Alternative: CSV Format

If your data is in CSV format:

```python
import pandas as pd

# Load from CSV
df = pd.read_csv('sensor_data.csv')

# Extract arrays
imu_data = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
mag_data = df[['mag_x', 'mag_y', 'mag_z']].values
positions = df[['x', 'y', 'floor']].values
activities = df['activity'].values

# Save as NumPy
np.save('imu_data.npy', imu_data)
np.save('magnetic_data.npy', mag_data)
np.save('positions.npy', positions)
np.save('activities.npy', activities)
```

## Expected Output

Training will create:

1. **checkpoints/** directory with:
   - `best_model.pt` - Best model based on validation loss
   - `checkpoint_epoch_*.pt` - Periodic checkpoints
   - `training_history.json` - Loss curves

2. **sensor_fusion_xlstm.onnx** - Exported model for Android

3. **normalizer_params.json** - Normalization parameters (needed for inference)

## Monitoring Training

Watch for:
- **Decreasing loss**: Both train and validation should decrease
- **No overfitting**: Validation loss should not increase while train loss decreases
- **Convergence**: Loss should stabilize after some epochs

## Troubleshooting

**Out of memory?**
- Reduce `batch_size` in config.yaml
- Reduce `hidden_size` or `num_layers`

**Not converging?**
- Increase `num_epochs`
- Adjust `learning_rate` (try 0.0001 or 0.01)
- Check your data for NaN values

**Poor accuracy?**
- Collect more training data
- Ensure ground truth labels are accurate
- Adjust feature extraction window size

## Next Steps

After training:

1. **Evaluate**: Check `training_history.json` for loss curves
2. **Test**: Use `inference.py` to test predictions
3. **Deploy**: Copy `sensor_fusion_xlstm.onnx` to Android app
4. **Integrate**: Follow Android integration guide in README.md
