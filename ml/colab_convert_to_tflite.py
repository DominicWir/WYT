# =============================================================================
# Colab: Export PyTorch xLSTM model to ONNX for Android
# 
# Instructions:
#   1. Upload: best_model.pt (from app/src/main/assets/) and xlstm_model.py (from ml/)
#   2. Run all cells
#   3. Download activity_model.onnx
#   4. Place it in app/src/main/assets/
# =============================================================================

# %% Cell 1: Install
!pip install torch onnx onnxruntime --quiet

# %% Cell 2: Verify uploads
import os, sys
required_files = ['best_model.pt', 'xlstm_model.py']
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    print("❌ Upload these:", missing)
    raise FileNotFoundError(missing)
print("✓ All files found")

# %% Cell 3: Load model
import torch, numpy as np
sys.path.insert(0, '.')
from xlstm_model import create_model

checkpoint = torch.load("best_model.pt", map_location="cpu")
config = checkpoint['config']
model = create_model(
    feature_dim=config['feature_dim'],
    config={'hidden_size': config.get('hidden_size', 256), 'num_layers': config.get('num_layers', 3)}
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ Model loaded (epoch {checkpoint['epoch']})")

# %% Cell 4: Activity-only wrapper (last timestep only)
class ActivityOnlyModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        _, activity, _ = self.model(x, return_uncertainty=False)
        return activity[:, -1, :]  # [batch, 5]

wrapped = ActivityOnlyModel(model)
wrapped.eval()
with torch.no_grad():
    print(f"✓ Output shape: {wrapped(torch.randn(1,50,69)).shape}")

# %% Cell 5: Export to ONNX
torch.onnx.export(
    wrapped, torch.randn(1, 50, 69), 'activity_model.onnx',
    export_params=True, opset_version=18, do_constant_folding=True,
    input_names=['input'], output_names=['activity']
)

# Re-save as single file (newer PyTorch may split into .onnx + .onnx.data)
import onnx
model_proto = onnx.load('activity_model.onnx', load_external_data=True)
onnx.save_model(model_proto, 'activity_model.onnx',
                save_as_external_data=False)
print(f"✓ ONNX exported ({os.path.getsize('activity_model.onnx')/1024/1024:.1f} MB, single file)")

# %% Cell 6: Verify
import onnxruntime as ort
session = ort.InferenceSession('activity_model.onnx')
inp = session.get_inputs()[0]
out = session.get_outputs()[0]
print(f"  Input:  {inp.name} {inp.shape} {inp.type}")
print(f"  Output: {out.name} {out.shape} {out.type}")

result = session.run(None, {'input': np.random.randn(1,50,69).astype(np.float32)})
labels = ['Stationary','Walking','Running','Stairs Up','Stairs Down']
print(f"  Test: {result[0]} → {labels[np.argmax(result[0])]}")

size_mb = os.path.getsize('activity_model.onnx') / 1024 / 1024
print(f"\n✅ activity_model.onnx ready! ({size_mb:.1f} MB)")

# %% Cell 7: Download
from google.colab import files
files.download('activity_model.onnx')
