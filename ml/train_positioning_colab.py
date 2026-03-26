"""
=============================================================================
WiFi Positioning Model Training for Google Colab
=============================================================================

QUICK START:
1. Upload this script + ujiindoorloc.zip to Colab
2. Run: !pip install torch numpy scipy scikit-learn onnx onnxruntime onnxscript pyyaml tqdm
3. Run: !python train_positioning_colab.py
4. Download: indoor_positioning.onnx + positioning_normalizer.json

The script is fully self-contained — no other files needed.
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import zipfile
import pandas as pd
import json
import os
import math
import warnings
from datetime import datetime
from tqdm import tqdm


# ============================================================================
# 1. DATA LOADING (UJIIndoorLoc)
# ============================================================================

def load_ujiindoorloc(zip_path: str, use_validation: bool = False) -> Dict[str, np.ndarray]:
    """Load UJIIndoorLoc WiFi fingerprint dataset from ZIP."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('UJIndoorLoc/trainingData.csv') as f:
            df_train = pd.read_csv(f)
        with z.open('UJIndoorLoc/validationData.csv') as f:
            df_val = pd.read_csv(f)
    
    df = df_val if use_validation else df_train
    print(f"Loaded {len(df)} samples from {'validation' if use_validation else 'training'} set")
    
    # Extract WiFi RSSI (520 columns)
    wifi_columns = [col for col in df.columns if col.startswith('WAP')]
    wifi_data = df[wifi_columns].values.astype(np.float32)
    wifi_data[wifi_data == 100] = -100  # Not-detected → very weak
    
    # Extract positions
    positions = np.column_stack([
        df['LONGITUDE'].values,
        df['LATITUDE'].values,
        df['FLOOR'].values
    ]).astype(np.float32)
    
    # Create synthetic IMU/mag (for pipeline compatibility)
    n = len(df)
    imu_data = np.random.randn(n, 6).astype(np.float32) * 0.5
    imu_data[:, 2] += 9.81
    mag_data = (np.random.randn(n, 3) * 5 + [20, 30, -40]).astype(np.float32)
    
    # Dummy activity labels
    activities = (df['SPACEID'].values % 5).astype(int)
    
    print(f"  WiFi shape: {wifi_data.shape}")
    print(f"  Positions shape: {positions.shape}")
    
    return {
        'imu': imu_data,
        'magnetic': mag_data,
        'wifi': wifi_data,
        'positions': positions,
        'activities': activities
    }


# ============================================================================
# 2. FEATURE EXTRACTION
# ============================================================================

class SensorFeatureExtractor:
    """Extract statistical features from sensor windows."""
    
    def __init__(self, window_size=50, stride=25, percentiles=[25.0, 50.0, 75.0]):
        self.window_size = window_size
        self.stride = stride
        self.percentiles = percentiles
    
    def extract_statistical_features(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.zeros(len(self.percentiles) + 4)
        features = [np.min(data, axis=0), np.max(data, axis=0),
                    np.mean(data, axis=0), np.std(data, axis=0)]
        for p in self.percentiles:
            features.append(np.percentile(data, p, axis=0))
        return np.concatenate([f.flatten() if f.ndim > 0 else [f] for f in features])
    
    def extract_window_features(self, imu, mag, wifi=None) -> np.ndarray:
        features = []
        # IMU features
        stat = self.extract_statistical_features(imu)
        acc_mag = np.linalg.norm(imu[:, :3], axis=1)
        gyro_mag = np.linalg.norm(imu[:, 3:], axis=1)
        features.append(np.concatenate([stat, [np.mean(acc_mag), np.std(acc_mag),
                                               np.mean(gyro_mag), np.std(gyro_mag)]]))
        # Magnetic features
        stat = self.extract_statistical_features(mag)
        mag_magnitude = np.linalg.norm(mag, axis=1)
        features.append(np.concatenate([stat, [np.mean(mag_magnitude), np.std(mag_magnitude)]]))
        # WiFi features
        if wifi is not None and len(wifi) > 0:
            all_rssi = wifi.flatten()
            all_rssi = all_rssi[~np.isnan(all_rssi)]
            if len(all_rssi) > 0:
                f = [np.min(all_rssi), np.max(all_rssi), np.mean(all_rssi), np.std(all_rssi)]
                for p in self.percentiles:
                    f.append(np.percentile(all_rssi, p))
                f.append(wifi.shape[1])
                features.append(np.array(f))
            else:
                features.append(np.zeros(len(self.percentiles) + 5))
        return np.concatenate(features)
    
    def extract_sequence_features(self, imu, mag, wifi=None):
        n = len(imu)
        features_list, timestamps = [], []
        for start in range(0, n - self.window_size + 1, self.stride):
            end = start + self.window_size
            f = self.extract_window_features(
                imu[start:end], mag[start:end],
                wifi[start:end] if wifi is not None else None)
            features_list.append(f)
            timestamps.append(start + self.window_size // 2)
        return np.array(features_list), np.array(timestamps)


# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

class SensorDataNormalizer:
    def __init__(self):
        self.scalers = {'imu': StandardScaler(), 'magnetic': StandardScaler(), 'wifi': StandardScaler()}
        self.fitted = False
    
    def fit(self, imu, mag, wifi=None):
        self.scalers['imu'].fit(imu)
        self.scalers['magnetic'].fit(mag)
        if wifi is not None:
            self.scalers['wifi'].fit(np.nan_to_num(wifi, nan=-100.0))
        self.fitted = True
    
    def transform(self, imu, mag, wifi=None):
        imu_n = self.scalers['imu'].transform(imu)
        mag_n = self.scalers['magnetic'].transform(mag)
        wifi_n = self.scalers['wifi'].transform(np.nan_to_num(wifi, nan=-100.0)) if wifi is not None else None
        return imu_n, mag_n, wifi_n
    
    def save(self, filepath):
        params = {
            'imu_mean': self.scalers['imu'].mean_.tolist(),
            'imu_scale': self.scalers['imu'].scale_.tolist(),
            'mag_mean': self.scalers['magnetic'].mean_.tolist(),
            'mag_scale': self.scalers['magnetic'].scale_.tolist(),
        }
        if hasattr(self.scalers['wifi'], 'mean_'):
            params['wifi_mean'] = self.scalers['wifi'].mean_.tolist()
            params['wifi_scale'] = self.scalers['wifi'].scale_.tolist()
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)


class SensorSequenceDataset(Dataset):
    def __init__(self, features, positions, activities, sequence_length=50):
        self.features = torch.FloatTensor(features)
        self.positions = torch.FloatTensor(positions)
        self.activities = torch.LongTensor(activities)
        self.sequence_length = sequence_length
        self.valid_indices = list(range(len(features) - sequence_length + 1))
    
    def __len__(self): return len(self.valid_indices)
    
    def __getitem__(self, idx):
        s = self.valid_indices[idx]
        e = s + self.sequence_length
        return {'features': self.features[s:e], 'positions': self.positions[s:e],
                'activities': self.activities[s:e]}


def handle_missing_data(imu, mag, wifi=None, method='interpolate'):
    def clean(arr, method):
        if method == 'zero': return np.nan_to_num(arr, nan=0.0)
        mask = np.isnan(arr)
        if mask.any():
            arr = arr.copy()
            for i in range(arr.shape[1]):
                if mask[:, i].any():
                    valid = ~mask[:, i]
                    if valid.any():
                        arr[mask[:, i], i] = np.interp(
                            np.where(mask[:, i])[0], np.where(valid)[0], arr[valid, i])
        return arr
    return clean(imu, method), clean(mag, method), clean(wifi, method) if wifi is not None else None


# ============================================================================
# 4. xLSTM MODEL
# ============================================================================

class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, head_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, head_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        self.ln_q = nn.LayerNorm(hidden_size)
        self.ln_k = nn.LayerNorm(head_size)
        self.ln_v = nn.LayerNorm(hidden_size)
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2: nn.init.xavier_uniform_(param)
            elif 'bias' in name: nn.init.zeros_(param)
    
    def forward(self, x, state=None):
        bs = x.size(0)
        if state is None:
            h = torch.zeros(bs, self.hidden_size, device=x.device)
            C = torch.zeros(bs, self.hidden_size, self.head_size, device=x.device)
            n = torch.ones(bs, self.hidden_size, device=x.device)
        else:
            h, C, n = state
        i = torch.sigmoid(self.W_i(x))
        f = torch.sigmoid(self.W_f(x))
        o = torch.sigmoid(self.W_o(x))
        q = self.ln_q(self.W_q(x))
        k = self.ln_k(self.W_k(x))
        v = self.ln_v(self.W_v(x))
        i_exp, f_exp = torch.exp(i), torch.exp(f)
        C_new = f_exp.unsqueeze(-1) * C + i_exp.unsqueeze(-1) * torch.bmm(v.unsqueeze(-1), k.unsqueeze(1))
        n_new = f_exp * n + i_exp
        h_new = o * (torch.bmm(C_new, k.unsqueeze(-1)).squeeze(-1) / (n_new + 1e-6))
        return h_new, (h_new, C_new, n_new)


class mLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, head_size=32, dropout=0.1):
        super().__init__()
        self.cell = mLSTMCell(input_size, hidden_size, head_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, state=None):
        outputs = []
        for t in range(x.size(1)):
            h, state = self.cell(x[:, t, :], state)
            outputs.append(h)
        outputs = self.layer_norm(self.dropout(torch.stack(outputs, dim=1)))
        return outputs, state


class SensorFusionXLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_size=256, num_layers=3, head_size=32,
                 num_attention_heads=8, dropout=0.1, output_position_dim=3, output_activity_dim=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_size), nn.LayerNorm(hidden_size),
            nn.ReLU(), nn.Dropout(dropout))
        self.lstm_layers = nn.ModuleList([
            mLSTMLayer(hidden_size, hidden_size, head_size, dropout)
            for _ in range(num_layers)])
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads,
                                               dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size // 2, output_position_dim))
        self.activity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size // 2, output_activity_dim))
        self.position_uncertainty = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, output_position_dim), nn.Softplus())
    
    def forward(self, x, return_uncertainty=False):
        x = self.input_proj(x)
        for layer in self.lstm_layers:
            x_out, _ = layer(x)
            x = x + x_out
        attn_out, _ = self.attention(x, x, x)
        x = self.attention_norm(x + attn_out)
        position = self.position_head(x)
        activity = self.activity_head(x)
        uncertainty = self.position_uncertainty(x) if return_uncertainty else None
        return position, activity, uncertainty


# ============================================================================
# 5. TRAINING
# ============================================================================

class MultiTaskLoss(nn.Module):
    def __init__(self, position_weight=1.0, activity_weight=0.0, use_uncertainty=False):
        super().__init__()
        self.position_weight = position_weight
        self.activity_weight = activity_weight
        self.position_loss = nn.MSELoss()
        self.activity_loss = nn.CrossEntropyLoss()
    
    def forward(self, pos_pred, pos_true, act_pred, act_true, uncertainty=None):
        pos_loss = self.position_loss(pos_pred, pos_true)
        bs, sl, nc = act_pred.shape
        act_loss = self.activity_loss(act_pred.reshape(-1, nc), act_true.reshape(-1))
        total = self.position_weight * pos_loss + self.activity_weight * act_loss
        return total, {'total': total.item(), 'position': pos_loss.item(), 'activity': act_loss.item()}


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Training on: {device}")
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.criterion = MultiTaskLoss(config.get('position_weight', 1.0),
                                        config.get('activity_weight', 0.0))
        self.optimizer = optim.AdamW(model.parameters(), lr=config.get('learning_rate', 1e-3),
                                      weight_decay=config.get('weight_decay', 1e-5))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                               factor=0.5, patience=5)
        self.best_val_loss = float('inf')
        self.epoch = 0
        self.history = {'train_loss': [], 'val_loss': [], 'train_position_loss': [],
                       'val_position_loss': []}
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints_positioning')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self):
        self.model.train()
        losses = {'total': 0, 'position': 0, 'activity': 0}
        n = 0
        for batch in tqdm(self.train_loader, desc=f'Epoch {self.epoch+1} [Train]'):
            feat = batch['features'].to(self.device)
            pos = batch['positions'].to(self.device)
            act = batch['activities'].to(self.device)
            self.optimizer.zero_grad()
            pos_p, act_p, _ = self.model(feat)
            loss, ld = self.criterion(pos_p, pos, act_p, act)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
            self.optimizer.step()
            for k in losses: losses[k] += ld[k]
            n += 1
        return {k: v/n for k, v in losses.items()}
    
    def validate(self):
        self.model.eval()
        losses = {'total': 0, 'position': 0, 'activity': 0}
        n = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Epoch {self.epoch+1} [Val]'):
                feat = batch['features'].to(self.device)
                pos = batch['positions'].to(self.device)
                act = batch['activities'].to(self.device)
                pos_p, act_p, _ = self.model(feat)
                _, ld = self.criterion(pos_p, pos, act_p, act)
                for k in losses: losses[k] += ld[k]
                n += 1
        return {k: v/n for k, v in losses.items()}
    
    def train(self, num_epochs):
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n")
        for epoch in range(num_epochs):
            self.epoch = epoch
            tl = self.train_epoch()
            vl = self.validate()
            self.history['train_loss'].append(tl['total'])
            self.history['val_loss'].append(vl['total'])
            self.history['train_position_loss'].append(tl['position'])
            self.history['val_position_loss'].append(vl['position'])
            self.scheduler.step(vl['total'])
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train: {tl['total']:.4f} (pos: {tl['position']:.4f})")
            print(f"  Val:   {vl['total']:.4f} (pos: {vl['position']:.4f})")
            if vl['total'] < self.best_val_loss:
                self.best_val_loss = vl['total']
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ New best model saved (val_loss: {self.best_val_loss:.4f})")
        print("\n✓ Training complete!")
        with open(os.path.join(self.checkpoint_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def save_checkpoint(self, filename):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, os.path.join(self.checkpoint_dir, filename))


# ============================================================================
# 6. ONNX EXPORT
# ============================================================================

def export_to_onnx(model, output_path, input_shape, opset_version=12):
    """Export trained model to ONNX."""
    model.eval()
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy, output_path,
        export_params=True, opset_version=opset_version,
        do_constant_folding=True,
        input_names=['sensor_features'],
        output_names=['position', 'activity_logits', 'uncertainty'],
        dynamic_axes={
            'sensor_features': {0: 'batch_size', 1: 'sequence_length'},
            'position': {0: 'batch_size', 1: 'sequence_length'},
            'activity_logits': {0: 'batch_size', 1: 'sequence_length'},
            'uncertainty': {0: 'batch_size', 1: 'sequence_length'}
        })
    print(f"✓ Exported to {output_path}")
    
    import onnx
    onnx.checker.check_model(onnx.load(output_path))
    print("✓ ONNX model verified")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("WIFI POSITIONING MODEL TRAINING (xLSTM + UJIIndoorLoc)")
    print("=" * 60 + "\n")
    
    # --- Configuration ---
    DATASET_PATH = "ujiindoorloc.zip"  # Upload this to Colab
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    SEQUENCE_LENGTH = 50
    HIDDEN_SIZE = 256
    NUM_LAYERS = 3
    LEARNING_RATE = 0.001
    
    # --- Load dataset ---
    print("Loading UJIIndoorLoc dataset...")
    dataset = load_ujiindoorloc(DATASET_PATH)
    
    imu = dataset['imu']
    mag = dataset['magnetic']
    wifi = dataset['wifi']
    positions = dataset['positions']
    activities = dataset['activities']
    
    # --- Preprocess ---
    print("\nPreprocessing...")
    imu_c, mag_c, wifi_c = handle_missing_data(imu, mag, wifi, method='interpolate')
    
    normalizer = SensorDataNormalizer()
    normalizer.fit(imu_c, mag_c, wifi_c)
    imu_n, mag_n, wifi_n = normalizer.transform(imu_c, mag_c, wifi_c)
    normalizer.save('positioning_normalizer.json')
    print("  ✓ Normalizer saved")
    
    # --- Extract features ---
    print("\nExtracting features...")
    extractor = SensorFeatureExtractor(window_size=50, stride=25, percentiles=[25.0, 50.0, 75.0])
    features, timestamps = extractor.extract_sequence_features(imu_n, mag_n, wifi_n)
    positions_aligned = positions[timestamps]
    activities_aligned = activities[timestamps]
    print(f"  Features: {features.shape}")
    
    # --- Train/val split ---
    np.random.seed(42)
    n = len(features)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    
    train_data = {'features': features[idx[:split]], 'positions': positions_aligned[idx[:split]],
                  'activities': activities_aligned[idx[:split]]}
    val_data = {'features': features[idx[split:]], 'positions': positions_aligned[idx[split:]],
                'activities': activities_aligned[idx[split:]]}
    
    print(f"  Train: {len(train_data['features'])}, Val: {len(val_data['features'])}")
    
    # --- DataLoaders ---
    train_ds = SensorSequenceDataset(train_data['features'], train_data['positions'],
                                      train_data['activities'], SEQUENCE_LENGTH)
    val_ds = SensorSequenceDataset(val_data['features'], val_data['positions'],
                                    val_data['activities'], SEQUENCE_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # --- Create model ---
    print("\nCreating xLSTM model...")
    feature_dim = features.shape[1]
    model = SensorFusionXLSTM(
        feature_dim=feature_dim, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        head_size=32, num_attention_heads=8, dropout=0.1,
        output_position_dim=3, output_activity_dim=5)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # --- Train ---
    config = {
        'feature_dim': feature_dim,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': 1e-5,
        'position_weight': 1.0,
        'activity_weight': 0.0,  # Position-only training
        'grad_clip': 1.0,
        'checkpoint_dir': 'checkpoints_positioning'
    }
    
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train(num_epochs=NUM_EPOCHS)
    
    # --- Export to ONNX ---
    print("\nExporting to ONNX...")
    
    # Load best model
    best_ckpt = torch.load('checkpoints_positioning/best_model.pt', map_location='cpu')
    model.load_state_dict(best_ckpt['model_state_dict'])
    
    export_to_onnx(model, 'indoor_positioning.onnx',
                   input_shape=(1, SEQUENCE_LENGTH, feature_dim))
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nFiles to download:")
    print(f"  1. indoor_positioning.onnx     — The trained model")
    print(f"  2. positioning_normalizer.json  — Normalizer parameters")
    print(f"\nCopy both files to: app/src/main/assets/")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
