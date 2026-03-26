"""
Training Script for 5-Class Gesture Detection on Google Colab

This script trains an activity recognition model with hand gesture detection
to reduce false positive step detections.

Classes:
  0: Walking (allow steps)
  1: Standing (block steps)
  2: Sitting (block steps)
  3: Hand Gestures (block steps)
  4: Typing/Fine Motor (block steps)

Optimized for Pixel 8 deployment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
from pathlib import Path
import json


def main():
    """Train 5-class gesture detection model on Colab GPU."""
    
    print("=" * 70)
    print("5-CLASS GESTURE DETECTION TRAINING (Google Colab)")
    print("=" * 70 + "\n")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠ WARNING: CUDA not available!")
        print("Please enable GPU: Runtime -> Change runtime type -> GPU")
        return
    
    device = 'cuda'
    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # ========================================
    # LOAD OPPORTUNITY DATASET WITH GESTURES
    # ========================================
    
    print("Loading Opportunity dataset with gesture labels...")
    
    # Add path to load_zip_datasets
    import sys
    sys.path.append('/content')
    
    from load_zip_datasets import load_dataset_from_zip
    
    data_path = "/content/opportunity+activity+recognition.zip"
    
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        print("Please upload opportunity+activity+recognition.zip to Colab")
        return
    
    # Load with gesture detection enabled
    dataset = load_dataset_from_zip(
        'opportunity',
        data_path,
        subjects=['S1', 'S2', 'S3'],  # 3 subjects for training
        sessions=['ADL1', 'ADL2', 'ADL3'],  # 3 sessions each
        include_gestures=True  # Enable 5-class labels
    )
    
    imu_data = dataset['imu']
    mag_data = dataset['magnetic']
    activities = dataset['activities']
    
    print(f"\nDataset loaded:")
    print(f"  IMU shape: {imu_data.shape}")
    print(f"  Magnetic shape: {mag_data.shape}")
    print(f"  Activities shape: {activities.shape}")
    print(f"  Total samples: {len(imu_data):,}\n")
    
    # Check class distribution
    unique, counts = np.unique(activities, return_counts=True)
    class_names = ['Walking', 'Standing', 'Sitting', 'Hand Gesture', 'Typing']
    print("Class distribution:")
    for cls, count in zip(unique, counts):
        if cls < len(class_names):
            pct = count / len(activities) * 100
            print(f"  {cls} ({class_names[cls]}): {count:,} samples ({pct:.1f}%)")
    
    # ========================================
    # FEATURE EXTRACTION
    # ========================================
    
    print("\nExtracting features...")
    
    from feature_extraction import SensorFeatureExtractor
    
    extractor = SensorFeatureExtractor(
        window_size=30,  # 1 second at 30 Hz
        stride=15,       # 50% overlap
        percentiles=[25, 50, 75]
    )
    
    features, timestamps = extractor.extract_sequence_features(
        imu_data, mag_data, None
    )
    
    activities_aligned = activities[timestamps]
    
    print(f"  Features extracted: {features.shape}")
    print(f"  Feature dimension: {features.shape[1]}\n")
    
    # ========================================
    # TRAIN/VAL SPLIT
    # ========================================
    
    print("Splitting data (80/20)...")
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        features, activities_aligned,
        test_size=0.2,
        random_state=42,
        stratify=activities_aligned
    )
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}\n")
    
    # ========================================
    # CREATE DATALOADERS
    # ========================================
    
    batch_size = 128  # Larger batch for GPU
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"DataLoaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}\n")
    
    # ========================================
    # CREATE MODEL (SimplifiedActivityModel)
    # ========================================
    
    print("Creating 5-class gesture detection model...")
    
    class GestureDetectionModel(nn.Module):
        """Simplified model for gesture detection (Pixel 8 optimized)."""
        
        def __init__(self, input_size, hidden_size=64, num_classes=5):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    model = GestureDetectionModel(
        input_size=features.shape[1],
        hidden_size=64,
        num_classes=5
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Input size: {features.shape[1]}")
    print(f"  Hidden size: 64")
    print(f"  Output classes: 5\n")
    
    # ========================================
    # TRAINING SETUP
    # ========================================
    
    # Class weights for imbalanced data
    class_counts = np.bincount(activities_aligned, minlength=5)  # Force 5 bins
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"Class weights (for imbalanced data):")
    for i, weight in enumerate(class_weights):
        if i < len(class_names):
            print(f"  {class_names[i]}: {weight:.3f}")
    print()
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # ========================================
    # TRAINING LOOP
    # ========================================
    
    print("Starting training...\n")
    print("=" * 70)
    
    num_epochs = 50
    best_val_loss = float('inf')
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, 'best_gesture_model.pt')
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
        print()
    
    print("=" * 70)
    print(f"\n✓ Training completed!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Best validation loss: {best_val_loss:.4f}\n")
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # ========================================
    # EXPORT TO JAVA (Pixel 8 Compatible)
    # ========================================
    
    print("Exporting model to Java...")
    
    # Load best model
    checkpoint = torch.load('best_gesture_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract weights
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.cpu().detach().numpy()
    
    # Create Java model code
    java_code = f'''package com.example.wytv2;

/**
 * 5-Class Gesture Detection Model
 * 
 * Classes:
 *   0: Walking (allow steps)
 *   1: Standing (block steps)
 *   2: Sitting (block steps)
 *   3: Hand Gestures (block steps)
 *   4: Typing/Fine Motor (block steps)
 * 
 * Trained on Opportunity dataset with gesture labels.
 * Validation Accuracy: {best_val_acc:.2f}%
 */
public class GestureDetectionModel {{
    
    private static final int INPUT_SIZE = {features.shape[1]};
    private static final int HIDDEN_SIZE = 64;
    private static final int NUM_CLASSES = 5;
    
    // Layer 1: Input -> Hidden
    private static final float[][] WEIGHTS_FC1 = {{{array_to_java_2d(weights['fc1.weight'])}}};
    private static final float[] BIAS_FC1 = {{{array_to_java_1d(weights['fc1.bias'])}}};
    
    // Layer 2: Hidden -> Output
    private static final float[][] WEIGHTS_FC2 = {{{array_to_java_2d(weights['fc2.weight'])}}};
    private static final float[] BIAS_FC2 = {{{array_to_java_1d(weights['fc2.bias'])}}};
    
    private static final String[] CLASS_NAMES = {{
        "Walking", "Standing", "Sitting", "Hand Gesture", "Typing"
    }};
    
    public static class PredictionResult {{
        public String activity;
        public float confidence;
        public float[] probabilities;
        
        public PredictionResult(String activity, float confidence, float[] probabilities) {{
            this.activity = activity;
            this.confidence = confidence;
            this.probabilities = probabilities;
        }}
    }}
    
    public static PredictionResult predictWithConfidence(float[] features) {{
        // Layer 1: fc1 + ReLU
        float[] hidden = new float[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {{
            float sum = BIAS_FC1[i];
            for (int j = 0; j < INPUT_SIZE; j++) {{
                sum += features[j] * WEIGHTS_FC1[i][j];
            }}
            hidden[i] = Math.max(0, sum);  // ReLU
        }}
        
        // Layer 2: fc2
        float[] logits = new float[NUM_CLASSES];
        for (int i = 0; i < NUM_CLASSES; i++) {{
            float sum = BIAS_FC2[i];
            for (int j = 0; j < HIDDEN_SIZE; j++) {{
                sum += hidden[j] * WEIGHTS_FC2[i][j];
            }}
            logits[i] = sum;
        }}
        
        // Softmax
        float[] probabilities = softmax(logits);
        
        // Get prediction
        int predictedClass = 0;
        float maxProb = probabilities[0];
        for (int i = 1; i < NUM_CLASSES; i++) {{
            if (probabilities[i] > maxProb) {{
                maxProb = probabilities[i];
                predictedClass = i;
            }}
        }}
        
        return new PredictionResult(
            CLASS_NAMES[predictedClass],
            maxProb,
            probabilities
        );
    }}
    
    private static float[] softmax(float[] logits) {{
        float[] probs = new float[logits.length];
        float maxLogit = logits[0];
        for (int i = 1; i < logits.length; i++) {{
            if (logits[i] > maxLogit) maxLogit = logits[i];
        }}
        
        float sum = 0.0f;
        for (int i = 0; i < logits.length; i++) {{
            probs[i] = (float) Math.exp(logits[i] - maxLogit);
            sum += probs[i];
        }}
        
        for (int i = 0; i < logits.length; i++) {{
            probs[i] /= sum;
        }}
        
        return probs;
    }}
}}
'''
    
    with open('GestureDetectionModel.java', 'w') as f:
        f.write(java_code)
    
    print("  ✓ Java model exported to GestureDetectionModel.java\n")
    
    # ========================================
    # SUMMARY
    # ========================================
    
    print("=" * 70)
    print("TRAINING COMPLETE - FILES CREATED")
    print("=" * 70)
    print("\n📁 Files to download:")
    print("  1. best_gesture_model.pt - PyTorch model")
    print("  2. GestureDetectionModel.java - Java model for Android")
    print("  3. training_history.json - Training metrics")
    print("\n📱 Next steps:")
    print("  1. Download GestureDetectionModel.java")
    print("  2. Replace SimplifiedActivityModel.java in Android project")
    print("  3. Update StepDetectionService to use 5-class model")
    print("  4. Test on Pixel 8")
    print("\n✓ Model ready for Pixel 8 deployment!")


def array_to_java_2d(arr):
    """Convert 2D numpy array to Java 2D array string."""
    rows = []
    for i, row in enumerate(arr):
        row_str = "{" + ", ".join([f"{x:.6f}f" for x in row]) + "}"
        # Add comma after each row except the last one
        if i < len(arr) - 1:
            row_str += ","
        rows.append(row_str)
    return "\n        ".join(rows)


def array_to_java_1d(arr):
    """Convert 1D numpy array to Java 1D array string."""
    return ", ".join([f"{x:.6f}f" for x in arr])


if __name__ == "__main__":
    main()
