"""
evaluate_positioning.py
=======================
Evaluates the saved WiFi positioning model (checkpoints/best_model.pt or
checkpoints_positioning/best_model.pt) on a held-out test split from UJIIndoorLoc.

Outputs:
  - Mean Euclidean Error (MEE) in metres
  - Mean Absolute Error per axis (x, y)
  - Floor detection accuracy (%)
  - Inference latency on this machine (ms) — a proxy for on-device latency

Run from the ml/ directory:
    python evaluate_positioning.py
"""

import numpy as np
import torch
import yaml
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split

from feature_extraction import SensorFeatureExtractor
from data_preprocessing import (
    SensorDataNormalizer,
    handle_missing_data
)
from inference import load_trained_model

UJIINDOORLOC_ZIP = "/Users/dominicwirasinha/Documents/IIT/Year 5/FYP Dev/ujiindoorloc.zip"
CHECKPOINT = "checkpoints/best_model.pt"   # adjust path if needed


def load_ujiindoorloc():
    from load_zip_datasets import load_dataset_from_zip
    dataset = load_dataset_from_zip('ujiindoorloc', UJIINDOORLOC_ZIP, use_validation=False)
    return dataset


def main():
    print("=" * 60)
    print("INDOOR POSITIONING MODEL EVALUATION (UJIIndoorLoc)")
    print("=" * 60 + "\n")

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # ── Load dataset ──────────────────────────────────────────────
    print("Loading UJIIndoorLoc dataset...")
    dataset = load_ujiindoorloc()
    imu_data   = dataset['imu']
    mag_data   = dataset['magnetic']
    wifi_data  = dataset['wifi']
    positions  = dataset['positions']   # (x, y, floor)
    activities = dataset['activities']

    print(f"  WiFi shape:      {wifi_data.shape}")
    print(f"  Positions shape: {positions.shape}")
    print(f"  Total samples:   {len(wifi_data)}")

    # ── Preprocess ────────────────────────────────────────────────
    print("\nPreprocessing...")
    imu_c, mag_c, wifi_c = handle_missing_data(
        imu_data, mag_data, wifi_data,
        method=config['preprocessing']['missing_data_method']
    )
    normalizer = SensorDataNormalizer()
    normalizer.fit(imu_c, mag_c, wifi_c)
    imu_n, mag_n, wifi_n = normalizer.transform(imu_c, mag_c, wifi_c)

    # ── Feature extraction ────────────────────────────────────────
    print("Extracting features...")
    extractor = SensorFeatureExtractor(
        window_size=config['feature_extraction']['window_size'],
        stride=config['feature_extraction']['stride'],
        percentiles=config['feature_extraction']['percentiles']
    )
    features, timestamps = extractor.extract_sequence_features(imu_n, mag_n, wifi_n)
    pos_aligned = positions[timestamps]

    print(f"  Feature matrix: {features.shape}")
    print(f"  Position range: x=[{pos_aligned[:,0].min():.1f}, {pos_aligned[:,0].max():.1f}]"
          f"  y=[{pos_aligned[:,1].min():.1f}, {pos_aligned[:,1].max():.1f}]"
          f"  floors={np.unique(pos_aligned[:,2].astype(int)).tolist()}")

    # ── Test split (15%) ─────────────────────────────────────────
    print("\nSplitting data (85% train+val / 15% test)...")
    _, X_test, _, y_test = train_test_split(
        features, pos_aligned, test_size=0.15, random_state=42
    )
    print(f"  Test samples: {len(X_test)}")

    # ── Load model ────────────────────────────────────────────────
    print("\nLoading model checkpoint...")
    model = load_trained_model(CHECKPOINT)
    model.eval()

    # ── Inference + latency measurement ───────────────────────────
    SEQ_LEN = config['training']['sequence_length']
    print(f"Running inference (sequence_length={SEQ_LEN})...")

    pred_xy     = []
    pred_floors = []
    true_xy     = []
    true_floors = []
    latencies   = []

    with torch.no_grad():
        for i in range(0, len(X_test) - SEQ_LEN + 1, SEQ_LEN):
            window = X_test[i:i + SEQ_LEN]
            gt     = y_test[i + SEQ_LEN - 1]   # ground truth for last timestep

            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

            t0 = time.perf_counter()
            pos_out, _, _ = model(x, return_uncertainty=False)
            t1 = time.perf_counter()

            pred = pos_out[0, -1, :].cpu().numpy()
            latencies.append((t1 - t0) * 1000)  # ms

            pred_xy.append(pred[:2])
            pred_floors.append(round(pred[2]))
            true_xy.append(gt[:2])
            true_floors.append(int(gt[2]))

    pred_xy     = np.array(pred_xy)
    true_xy     = np.array(true_xy)
    pred_floors = np.array(pred_floors)
    true_floors = np.array(true_floors)

    # ── Compute metrics ───────────────────────────────────────────
    euclidean_errors = np.sqrt(np.sum((pred_xy - true_xy) ** 2, axis=1))
    mee    = euclidean_errors.mean()
    mee_50 = np.percentile(euclidean_errors, 50)
    mee_75 = np.percentile(euclidean_errors, 75)
    mee_90 = np.percentile(euclidean_errors, 90)

    mae_x = np.abs(pred_xy[:, 0] - true_xy[:, 0]).mean()
    mae_y = np.abs(pred_xy[:, 1] - true_xy[:, 1]).mean()

    floor_acc = (pred_floors == true_floors).mean() * 100
    mean_lat  = np.mean(latencies)

    print("\n" + "=" * 60)
    print("POSITIONING EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Mean Euclidean Error (MEE):   {mee:.2f} m")
    print(f"  Median Error:                 {mee_50:.2f} m")
    print(f"  75th percentile error:        {mee_75:.2f} m")
    print(f"  90th percentile error:        {mee_90:.2f} m")
    print(f"  MAE x-axis:                   {mae_x:.2f} m")
    print(f"  MAE y-axis:                   {mae_y:.2f} m")
    print(f"  Floor detection accuracy:     {floor_acc:.1f}%")
    print(f"  Mean inference latency:       {mean_lat:.1f} ms")
    print(f"  Test samples evaluated:       {len(euclidean_errors)}")

    # ── Save results ──────────────────────────────────────────────
    results = {
        "mee_m": round(mee, 3),
        "median_error_m": round(float(mee_50), 3),
        "p75_error_m": round(float(mee_75), 3),
        "p90_error_m": round(float(mee_90), 3),
        "mae_x_m": round(float(mae_x), 3),
        "mae_y_m": round(float(mae_y), 3),
        "floor_accuracy_pct": round(float(floor_acc), 1),
        "mean_latency_ms": round(float(mean_lat), 1),
        "test_samples": len(euclidean_errors)
    }
    with open("positioning_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✓ Results saved to positioning_eval_results.json")


if __name__ == "__main__":
    main()
