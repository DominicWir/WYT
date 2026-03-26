#!/bin/bash

# Restart Training Script with GPU Support
# This script stops any running training and restarts with GPU (MPS) acceleration

echo "=============================================="
echo "RESTARTING TRAINING WITH GPU ACCELERATION"
echo "=============================================="
echo ""

# Stop any existing training processes
echo "1. Checking for running training processes..."
PIDS=$(ps aux | grep -E "train_activity|train_positioning" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "   ✓ No training processes running"
else
    echo "   Found processes: $PIDS"
    echo "   Stopping them..."
    kill $PIDS 2>/dev/null
    sleep 2
    echo "   ✓ Processes stopped"
fi

echo ""
echo "2. Checking GPU availability..."

# Activate virtual environment if it exists
if [ -d "../.venv" ]; then
    source ../.venv/bin/activate
    echo "   ✓ Using virtual environment"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "   ✓ Using virtual environment"
fi

# Check MPS availability
python3 << 'EOF'
import torch
print(f"   CUDA available: {torch.cuda.is_available()}")
if hasattr(torch.backends, 'mps'):
    print(f"   MPS available: {torch.backends.mps.is_available()}")
    print(f"   MPS built: {torch.backends.mps.is_built()}")
    if torch.backends.mps.is_available():
        print("   ✓ GPU (MPS) will be used for training!")
    else:
        print("   ⚠ MPS not available, will use CPU")
else:
    print("   ⚠ MPS not supported in this PyTorch version")
EOF

echo ""
echo "3. Starting GPU-accelerated training..."
echo "   Training will automatically use MPS if available"
echo "   Log file: activity_training_gpu.log"
echo ""
echo "=============================================="
echo ""

# Start training with GPU
# The train_activity_gpu.py script explicitly forces MPS usage
nohup python3 train_activity_gpu.py > activity_training_gpu.log 2>&1 &

TRAIN_PID=$!
echo "✓ Training started with PID: $TRAIN_PID"
echo ""
echo "Monitor progress with:"
echo "  tail -f activity_training_gpu.log"
echo ""
echo "Check GPU usage with:"
echo "  sudo powermetrics --samplers gpu_power -i 1000 -n 1"
echo ""
