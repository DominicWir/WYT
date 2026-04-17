"""
evaluate_activity.py
====================
Evaluates the saved activity recognition model (checkpoints_activity/best_model.pt)
on a held-out test split from the Opportunity dataset.

Outputs:
  - Overall accuracy
  - Per-class Precision / Recall / F1
  - 5x5 Confusion matrix
  - Class distribution

Run from the ml/ directory:
    python evaluate_activity.py
"""

import numpy as np
import torch
import yaml
import json
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.model_selection import train_test_split

from feature_extraction import SensorFeatureExtractor
from data_preprocessing import (
    SensorDataNormalizer,
    handle_missing_data
)
from xlstm_model import create_model
from inference import load_trained_model

CLASS_NAMES = ["Stationary", "Walking", "Running", "Stairs Up", "Stairs Down"]


def load_opportunity_data():
    """Load Opportunity dataset using the existing zip loader."""
    from load_zip_datasets import load_dataset_from_zip
    data_path = "/Users/dominicwirasinha/Documents/IIT/Year 5/FYP Dev/opportunity+activity+recognition.zip"
    dataset = load_dataset_from_zip('opportunity', data_path, use_validation=False)
    return dataset


def main():
    print("=" * 60)
    print("ACTIVITY RECOGNITION MODEL EVALUATION")
    print("=" * 60 + "\n")

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # ── Load dataset ──────────────────────────────────────────────
    print("Loading Opportunity dataset...")
    dataset = load_opportunity_data()
    imu_data  = dataset['imu']
    mag_data  = dataset['magnetic']
    wifi_data = dataset['wifi']
    activities = dataset['activities']

    print(f"  IMU shape:        {imu_data.shape}")
    print(f"  Activities shape: {activities.shape}")

    # ── Preprocess ────────────────────────────────────────────────
    print("\nPreprocessing...")
    imu_clean, mag_clean, wifi_clean = handle_missing_data(
        imu_data, mag_data, wifi_data,
        method=config['preprocessing']['missing_data_method']
    )

    normalizer = SensorDataNormalizer()
    normalizer.fit(imu_clean, mag_clean, wifi_clean)
    imu_norm, mag_norm, wifi_norm = normalizer.transform(imu_clean, mag_clean, wifi_clean)

    # ── Feature extraction ────────────────────────────────────────
    print("Extracting features...")
    extractor = SensorFeatureExtractor(
        window_size=config['feature_extraction']['window_size'],
        stride=config['feature_extraction']['stride'],
        percentiles=config['feature_extraction']['percentiles']
    )
    features, timestamps = extractor.extract_sequence_features(imu_norm, mag_norm, wifi_norm)
    labels = activities[timestamps].astype(int)

    print(f"  Feature matrix: {features.shape}")

    # ── Class distribution ────────────────────────────────────────
    print("\nClass distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
        print(f"  {name}: {cnt} samples ({100*cnt/len(labels):.1f}%)")

    # ── Train / Val / Test split (70/15/15) ───────────────────────
    print("\nSplitting data (70/15/15)...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        features, labels, test_size=0.15, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15/0.85, random_state=42, stratify=y_trainval
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Load model ────────────────────────────────────────────────
    print("\nLoading best_model.pt ...")
    model = load_trained_model("checkpoints_activity/best_model.pt")
    model.eval()

    # ── Inference on test set ─────────────────────────────────────
    SEQ_LEN = config['training']['sequence_length']
    print(f"Running inference (sequence_length={SEQ_LEN})...")

    all_preds, all_true = [], []
    with torch.no_grad():
        for i in range(0, len(X_test) - SEQ_LEN + 1, SEQ_LEN):
            window = X_test[i:i + SEQ_LEN]
            label  = y_test[i + SEQ_LEN - 1]  # label of last timestep in window

            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            pos_out, act_logits, _ = model(x, return_uncertainty=False)
            pred = torch.argmax(act_logits[0, -1, :]).item()

            all_preds.append(pred)
            all_true.append(label)

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    # ── Results ───────────────────────────────────────────────────
    acc = accuracy_score(all_true, all_preds) * 100
    print("\n" + "=" * 60)
    print(f"OVERALL ACCURACY: {acc:.2f}%")
    print("=" * 60)

    print("\nPer-class Classification Report:")
    print(classification_report(all_true, all_preds, target_names=CLASS_NAMES, digits=3))

    print("Confusion Matrix (rows=True, cols=Predicted):")
    cm = confusion_matrix(all_true, all_preds, labels=list(range(len(CLASS_NAMES))))
    header = "         " + "  ".join(f"{n[:8]:>8}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i][:8]:>8} " + "  ".join(f"{v:>8}" for v in row))

    # ── Save results ──────────────────────────────────────────────
    results = {
        "overall_accuracy_pct": round(acc, 2),
        "classification_report": classification_report(
            all_true, all_preds, target_names=CLASS_NAMES, output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "test_samples": len(all_true)
    }
    with open("activity_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✓ Results saved to activity_eval_results.json")


if __name__ == "__main__":
    main()
