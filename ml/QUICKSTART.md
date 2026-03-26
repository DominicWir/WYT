# 🚀 Quick Start: Train Your Model

## Step 1: Activate Virtual Environment

```bash
cd /Users/dominicwirasinha/AndroidStudioProjects/WYTV2
source .venv/bin/activate
```

## Step 2: Navigate to ML Directory

```bash
cd ml
```

## Step 3: Run Training

```bash
python train_custom_dataset.py
```

That's it! The training will:
1. ✓ Load UJIIndoorLoc dataset (19,937 samples)
2. ✓ Extract features from Wi-Fi data
3. ✓ Train xLSTM model
4. ✓ Save best model to `checkpoints/best_model.pt`
5. ✓ Export to `sensor_fusion_xlstm.onnx`

## What to Expect

**Training Progress:**
- You'll see progress bars for each epoch
- Training loss and validation loss will be displayed
- Best model is saved automatically

**Training Time:**
- ~10-30 minutes depending on your CPU
- Default: 100 epochs (adjust in `config.yaml`)

**Output Files:**
- `checkpoints/best_model.pt` - Best trained model
- `checkpoints/training_history.json` - Loss curves
- `sensor_fusion_xlstm.onnx` - For Android deployment
- `normalizer_params.json` - Normalization parameters

## Monitor Training

Watch for:
- ✓ **Decreasing loss** - Both train and val should go down
- ✓ **Convergence** - Loss stabilizes after some epochs
- ⚠️ **Overfitting** - If val loss increases while train decreases

## Adjust Settings

Edit `config.yaml`:

```yaml
training:
  num_epochs: 50      # Reduce for faster testing
  batch_size: 16      # Reduce if out of memory
  learning_rate: 0.001
```

## Troubleshooting

**Out of memory?**
```yaml
# In config.yaml
training:
  batch_size: 8  # or even 4
model:
  hidden_size: 128  # reduce from 256
```

**Training too slow?**
```yaml
training:
  num_epochs: 20  # reduce for testing
```

## After Training

1. **Check results**: `cat checkpoints/training_history.json`
2. **Test model**: Use `inference.py` for predictions
3. **Deploy**: Copy `sensor_fusion_xlstm.onnx` to Android app

## Next: Deploy to Android

See `README.md` for Android integration instructions.
