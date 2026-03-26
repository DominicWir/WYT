# Training Started! 🚀

## Current Configuration

**Dataset**: Opportunity Activity Recognition
- **Subjects**: S1, S2, S3 (3 subjects)
- **Sessions**: ADL1, ADL2, ADL3 (daily activities)
- **Expected samples**: ~150,000-200,000

**Model Settings**:
- Hidden size: 256
- Layers: 3
- Sequence length: 50 (increased for better temporal context)

**Training**:
- Epochs: 30 (initial test - increase to 100 for full training)
- Batch size: 32
- Learning rate: 0.001
- Activity weight: 1.5 (prioritized over position)
- Position weight: 0.5 (since positions are estimated)

## What's Happening

The training script is now:
1. ✓ Loading Opportunity dataset (3 subjects × 3 sessions)
2. ✓ Extracting real IMU features (accelerometer + gyroscope)
3. ✓ Extracting real magnetic field data
4. ✓ Processing activity labels
5. → Training xLSTM model...

## Monitor Progress

Watch for:
- **Loading progress**: Each subject-session combination
- **Feature extraction**: Statistical features from sensor windows
- **Training epochs**: Loss should decrease
- **Best model saves**: Automatically saved when validation improves

## Expected Timeline

- **Data loading**: ~2-5 minutes
- **Feature extraction**: ~3-5 minutes
- **Training (30 epochs)**: ~15-30 minutes
- **Total**: ~20-40 minutes

## Output Files

Will be created in `checkpoints/`:
- `best_model.pt` - Best model checkpoint
- `training_history.json` - Loss curves
- `checkpoint_epoch_*.pt` - Periodic saves

After training:
- `sensor_fusion_xlstm.onnx` - For Android deployment
- `normalizer_params.json` - Normalization parameters

## Next Steps

After this initial 30-epoch training:
1. Check `training_history.json` for loss curves
2. If results look good, increase to 100 epochs
3. Test with `inference.py`
4. Deploy ONNX model to Android

## Adjust Settings

If needed, edit `config.yaml`:
- Reduce `batch_size` if out of memory
- Increase `num_epochs` for better accuracy
- Adjust `activity_weight` vs `position_weight`
