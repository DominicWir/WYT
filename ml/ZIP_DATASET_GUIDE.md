# Training with ZIP Datasets - Quick Guide

## ✅ Working Dataset: UJIIndoorLoc

Your **UJIIndoorLoc** dataset is ready to use!

### What's in the Dataset

- **520 Wi-Fi Access Points** with RSSI values
- **19,937 training samples**
- **1,111 validation samples**
- **Position labels**: Longitude, Latitude, Floor, Building
- **3 buildings, 4-5 floors each**

### How to Train

The training script is already configured for you:

```bash
cd /Users/dominicwirasinha/AndroidStudioProjects/WYTV2/ml
python train_custom_dataset.py
```

### What the Script Does

1. ✓ Loads UJIIndoorLoc from ZIP
2. ✓ Extracts Wi-Fi RSSI data (520 access points)
3. ✓ Uses real position labels (longitude, latitude, floor)
4. ✓ Creates synthetic IMU/magnetic data (since dataset doesn't have it)
5. ✓ Trains xLSTM model for indoor positioning

### Important Notes

> [!WARNING]
> **UJIIndoorLoc only contains Wi-Fi data, not IMU sensors.**
> 
> The loader creates synthetic IMU and magnetic data for compatibility.
> For real-world use, you should collect actual IMU data from your Android app.

### Configuration

Edit `config.yaml` to adjust:

```yaml
feature_extraction:
  window_size: 50  # Adjust based on your needs

training:
  batch_size: 32   # Reduce if out of memory
  num_epochs: 50   # Increase for better accuracy
```

## ⚠️ Opportunity Dataset Issue

The **opportunity+activity+recognition.zip** file appears to be corrupted or in an incompatible format.

### Solutions

1. **Re-download** from official source:
   - https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition

2. **Alternative**: Use UJIIndoorLoc for now and collect your own IMU data from the Android app

3. **Check file integrity**: The file shows as "Zip archive" but Python can't read it properly

## 📊 Expected Training Results

With UJIIndoorLoc dataset:
- **Position accuracy**: ~5-10 meters (Wi-Fi fingerprinting)
- **Floor accuracy**: ~95%+
- **Training time**: ~10-30 minutes (depending on epochs)

## Next Steps

1. **Run training**: `python train_custom_dataset.py`
2. **Monitor progress**: Watch loss decrease in terminal
3. **Check results**: Review `checkpoints/training_history.json`
4. **Export model**: ONNX file will be created automatically
5. **Deploy**: Copy ONNX to Android app

## Collecting Your Own Data

For best results with your WYT V2 app:

1. **Export sensor data** from your Android app:
   - IMU (accelerometer + gyroscope)
   - Magnetic field
   - Wi-Fi RSSI (if available)
   - Ground truth positions

2. **Save as NumPy arrays**:
   ```python
   np.save('my_imu_data.npy', imu_data)
   np.save('my_mag_data.npy', mag_data)
   np.save('my_positions.npy', positions)
   ```

3. **Update training script** to use your data instead of UJIIndoorLoc

## Files Created

- [`load_zip_datasets.py`](file:///Users/dominicwirasinha/AndroidStudioProjects/WYTV2/ml/load_zip_datasets.py) - ZIP dataset loader
- [`train_custom_dataset.py`](file:///Users/dominicwirasinha/AndroidStudioProjects/WYTV2/ml/train_custom_dataset.py) - Updated training script
- [`DATASET_GUIDE.md`](file:///Users/dominicwirasinha/AndroidStudioProjects/WYTV2/ml/DATASET_GUIDE.md) - Detailed guide
