package com.example.wytv2.ml;

/**
 * Data point representing a single step detection event with sensor data and
 * ground truth.
 * Used for collecting training data for continuous learning.
 */
public class StepDataPoint {
    // Timestamp
    public long timestamp;

    // Raw sensor data (3 values each)
    public float[] accelerometer; // x, y, z
    public float[] gyroscope; // x, y, z
    public float[] magnetometer; // x, y, z

    // Extracted features (69 dimensions)
    public float[] features;

    // Ground truth label
    public boolean isActualStep; // True if this was a real step

    // Detection metadata
    public float threshold; // Threshold used at detection time
    public String activityLabel; // "Walking", "Standing", "Hand Gesture", etc.
    public float confidence; // Model confidence (0-1)
    public String source; // "calibration" or "normal"

    // Training metadata
    public boolean usedForTraining; // Has this been used in training?

    public StepDataPoint() {
        this.timestamp = System.currentTimeMillis();
        this.usedForTraining = false;
    }

    /**
     * Create a data point with all sensor data.
     */
    public StepDataPoint(float[] accel, float[] gyro, float[] mag,
            float[] features, boolean isActualStep,
            float threshold, String activity, float confidence,
            String source) {
        this();
        this.accelerometer = accel;
        this.gyroscope = gyro;
        this.magnetometer = mag;
        this.features = features;
        this.isActualStep = isActualStep;
        this.threshold = threshold;
        this.activityLabel = activity;
        this.confidence = confidence;
        this.source = source;
    }

    /**
     * Validate that all required data is present.
     */
    public boolean isValid() {
        return accelerometer != null && accelerometer.length == 3 &&
                gyroscope != null && gyroscope.length == 3 &&
                magnetometer != null && magnetometer.length == 3 &&
                features != null && features.length == 69;
    }

    /**
     * Get label as integer (0 = not step, 1 = step).
     */
    public int getLabelAsInt() {
        return isActualStep ? 1 : 0;
    }
}
