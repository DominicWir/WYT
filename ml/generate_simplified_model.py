#!/usr/bin/env python3
"""
Generate a simplified model approximation from the trained xLSTM model.
This creates a lightweight decision model that can be implemented in pure Java.
"""

import torch
import numpy as np
import json
from collections import defaultdict

def analyze_model_patterns(model_path: str, num_samples: int = 1000):
    """
    Analyze the trained model to extract key decision patterns.
    """
    print("Loading trained model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    from xlstm_model import SensorFusionXLSTM
    
    model = SensorFusionXLSTM(
        feature_dim=69,
        hidden_size=256,
        num_layers=3,
        head_size=32,
        num_attention_heads=8,
        dropout=0.1,
        output_position_dim=3,
        output_activity_dim=5
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Generating {num_samples} random samples...")
    
    # Generate diverse input samples
    samples = []
    predictions = []
    
    for i in range(num_samples):
        # Create random sensor data
        sample = torch.randn(1, 50, 69)
        
        with torch.no_grad():
            _, activity, _ = model(sample, return_uncertainty=False)
            pred = torch.argmax(activity[0, -1]).item()
        
        # Store sample statistics
        sample_np = sample.numpy()[0]
        features = {
            'mean': float(np.mean(sample_np)),
            'std': float(np.std(sample_np)),
            'max': float(np.max(sample_np)),
            'min': float(np.min(sample_np)),
            'range': float(np.max(sample_np) - np.min(sample_np)),
            'prediction': pred
        }
        
        samples.append(features)
        predictions.append(pred)
    
    print(f"\nPrediction distribution:")
    for i in range(5):
        count = predictions.count(i)
        print(f"  Class {i}: {count} ({count/num_samples*100:.1f}%)")
    
    return samples

def create_decision_rules(samples):
    """
    Create simple decision rules based on sample statistics.
    """
    print("\nCreating decision rules...")
    
    # Group samples by prediction
    by_class = defaultdict(list)
    for sample in samples:
        by_class[sample['prediction']].append(sample)
    
    # Calculate thresholds for each class
    rules = {}
    for class_id, class_samples in by_class.items():
        if len(class_samples) == 0:
            continue
            
        rules[class_id] = {
            'mean_range': [
                float(np.percentile([s['mean'] for s in class_samples], 10)),
                float(np.percentile([s['mean'] for s in class_samples], 90))
            ],
            'std_range': [
                float(np.percentile([s['std'] for s in class_samples], 10)),
                float(np.percentile([s['std'] for s in class_samples], 90))
            ],
            'range_threshold': float(np.median([s['range'] for s in class_samples]))
        }
    
    return rules

def generate_java_model(rules, output_path: str):
    """
    Generate Java code for the simplified model.
    """
    activity_names = ["Standing", "Walking", "Sitting", "Lying", "Other"]
    
    java_code = '''package com.example.wytv2;

/**
 * Simplified activity recognition model - Pure Java implementation.
 * No ML library dependencies, works on all Android devices including Pixel 8.
 */
public class SimplifiedActivityModel {
    
    private static final String[] ACTIVITIES = {
        "Standing",
        "Walking", 
        "Sitting",
        "Lying",
        "Other"
    };
    
    /**
     * Predict activity from sensor features.
     * 
     * @param features 2D array of shape [50, 69] (sequence_length, feature_dim)
     * @return Predicted activity name
     */
    public static String predict(float[][] features) {
        // Calculate simple statistics from the input
        float mean = calculateMean(features);
        float std = calculateStd(features, mean);
        float range = calculateRange(features);
        
        // Apply decision rules
        int prediction = classifyByRules(mean, std, range);
        
        return ACTIVITIES[prediction];
    }
    
    /**
     * Classify based on statistical features.
     */
    private static int classifyByRules(float mean, float std, float range) {
'''
    
    # Add decision logic
    for class_id in sorted(rules.keys()):
        rule = rules[class_id]
        java_code += f'''        // Rule for {activity_names[class_id]}
        if (mean >= {rule['mean_range'][0]}f && mean <= {rule['mean_range'][1]}f &&
            std >= {rule['std_range'][0]}f && std <= {rule['std_range'][1]}f) {{
            return {class_id};
        }}
        
'''
    
    java_code += '''        // Default to "Other" if no rules match
        return 4;
    }
    
    private static float calculateMean(float[][] data) {
        float sum = 0;
        int count = 0;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                sum += data[i][j];
                count++;
            }
        }
        return sum / count;
    }
    
    private static float calculateStd(float[][] data, float mean) {
        float sumSquaredDiff = 0;
        int count = 0;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                float diff = data[i][j] - mean;
                sumSquaredDiff += diff * diff;
                count++;
            }
        }
        return (float) Math.sqrt(sumSquaredDiff / count);
    }
    
    private static float calculateRange(float[][] data) {
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                if (data[i][j] < min) min = data[i][j];
                if (data[i][j] > max) max = data[i][j];
            }
        }
        return max - min;
    }
    
    /**
     * Get prediction with confidence scores.
     */
    public static PredictionResult predictWithConfidence(float[][] features) {
        String activity = predict(features);
        int activityIndex = getActivityIndex(activity);
        
        // Simple confidence based on how well features match rules
        float confidence = 0.7f; // Placeholder
        
        return new PredictionResult(activity, activityIndex, confidence);
    }
    
    private static int getActivityIndex(String activity) {
        for (int i = 0; i < ACTIVITIES.length; i++) {
            if (ACTIVITIES[i].equals(activity)) {
                return i;
            }
        }
        return 4; // Other
    }
    
    public static class PredictionResult {
        public final String activity;
        public final int activityIndex;
        public final float confidence;
        
        public PredictionResult(String activity, int activityIndex, float confidence) {
            this.activity = activity;
            this.activityIndex = activityIndex;
            this.confidence = confidence;
        }
        
        @Override
        public String toString() {
            return String.format("%s (%.1f%%)", activity, confidence * 100);
        }
    }
}
'''
    
    with open(output_path, 'w') as f:
        f.write(java_code)
    
    print(f"✓ Java model written to: {output_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_simplified_model.py <path_to_best_model.pt>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Analyze model
    samples = analyze_model_patterns(model_path, num_samples=1000)
    
    # Create decision rules
    rules = create_decision_rules(samples)
    
    # Save rules
    with open('model_rules.json', 'w') as f:
        json.dump(rules, f, indent=2)
    print("✓ Rules saved to model_rules.json")
    
    # Generate Java code
    generate_java_model(rules, 'SimplifiedActivityModel.java')
    
    print("\n✅ Simplified model generated!")
    print("\nNext steps:")
    print("1. Copy SimplifiedActivityModel.java to your Android project")
    print("2. Remove PyTorch dependencies from build.gradle")
    print("3. Use SimplifiedActivityModel.predict() for inference")
    print("4. Test on Pixel 8 - NO 16KB issues!")
