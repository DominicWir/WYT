# GPU Training Guide

## Your Hardware

**GPUs Detected:**
- Intel UHD Graphics 630
- **Radeon Pro 560X** (dedicated GPU)
- Metal 3 support ✅

## Speed Comparison

### CPU Training (Current):
- **Time per epoch**: ~15-18 minutes
- **Total time (30 epochs)**: ~8-10 hours
- **Current progress**: Epoch 6/30

### GPU Training (Estimated):
- **Time per epoch**: ~1-3 minutes (5-15x faster)
- **Total time (30 epochs)**: ~30 minutes - 1.5 hours
- **Speedup**: **5-15x faster** depending on batch size

## Why GPU is Faster

✅ **Parallel processing**: GPUs have thousands of cores
✅ **Matrix operations**: Perfect for neural networks
✅ **Batch processing**: Process 32 samples simultaneously
✅ **Memory bandwidth**: Faster data transfer

## How to Enable GPU

### Option 1: Use MPS (Metal Performance Shaders) - macOS

PyTorch supports Apple Silicon and AMD GPUs via MPS backend.

**Check if available:**
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
```

**Enable in training script:**
```python
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
```

### Option 2: Stop Current Training & Restart with GPU

1. **Stop current training**: `pkill -f train_activity`
2. **Run GPU-enabled script**: `python train_activity_gpu.py`
3. **Monitor**: Much faster progress!

## Recommendation

**For your setup (Radeon Pro 560X):**
- Expected speedup: **8-12x faster**
- Epoch time: **1.5-2 minutes** (vs 15-18 minutes)
- Total time: **45-60 minutes** (vs 8-10 hours)

**Worth it?** 
- ✅ **YES!** Save 7-9 hours
- Current training will take ~7 more hours
- GPU training will finish in ~1 hour

## Next Steps

1. I can create a GPU-enabled training script
2. Stop current CPU training
3. Restart with GPU acceleration
4. Complete in ~1 hour instead of ~7 hours

**Want me to set this up?**
