package com.example.wytv2.ml;

import android.content.Context;
import android.util.Log;

import java.util.List;

/**
 * Service for collecting step detection data for continuous learning.
 * Collects labeled data during calibration and unlabeled data during normal
 * use.
 */
public class StepDataCollector {
    private static final String TAG = "StepDataCollector";

    private StepDataDatabase database;
    private boolean collectingEnabled = true;
    private int samplesCollected = 0;

    public StepDataCollector(Context context) {
        this.database = new StepDataDatabase(context);
        Log.d(TAG, "StepDataCollector initialized");
    }

    /**
     * Enable or disable data collection.
     */
    public void setCollectingEnabled(boolean enabled) {
        this.collectingEnabled = enabled;
        Log.d(TAG, "Data collection " + (enabled ? "enabled" : "disabled"));
    }

    public boolean isCollectingEnabled() {
        return collectingEnabled;
    }

    /**
     * Record a step candidate with all sensor data and ground truth.
     * 
     * @param accel         Accelerometer data [x, y, z]
     * @param gyro          Gyroscope data [x, y, z]
     * @param mag           Magnetometer data [x, y, z]
     * @param features      Extracted 69-dimensional feature vector
     * @param wasActualStep Ground truth label (true if real step)
     * @param threshold     Threshold used at detection time
     * @param activity      Activity label from ML model
     * @param confidence    Model confidence (0-1)
     * @param source        "calibration" or "normal"
     */
    public void recordStepCandidate(
            float[] accel, float[] gyro, float[] mag,
            float[] features, boolean wasActualStep,
            float threshold, String activity, float confidence,
            String source) {

        if (!collectingEnabled) {
            return;
        }

        // Create data point
        StepDataPoint dataPoint = new StepDataPoint(
                accel, gyro, mag, features, wasActualStep,
                threshold, activity, confidence, source);

        // Validate before inserting
        if (!dataPoint.isValid()) {
            Log.w(TAG, "Invalid data point, skipping");
            return;
        }

        // Insert into database
        long id = database.insert(dataPoint);
        if (id > 0) {
            samplesCollected++;
            if (samplesCollected % 10 == 0) {
                Log.d(TAG, "Collected " + samplesCollected + " samples total");
            }
        }
    }

    /**
     * Get total number of collected samples.
     */
    public int getCollectedSamplesCount() {
        return database.getCount();
    }

    /**
     * Get number of unused training samples.
     */
    public int getUnusedSamplesCount() {
        return database.getUnusedCount();
    }

    /**
     * Get unused training data (not yet used for model training).
     */
    public List<StepDataPoint> getUnusedTrainingData(int limit) {
        return database.getUnusedData(limit);
    }

    /**
     * Mark data points as used for training.
     */
    public void markDataAsUsed(List<StepDataPoint> dataPoints) {
        database.markAsUsed(dataPoints);
    }

    /**
     * Clear all collected data (for privacy/testing).
     */
    public void clearAllData() {
        database.clearAll();
        samplesCollected = 0;
        Log.d(TAG, "All data cleared");
    }

    /**
     * Get statistics about collected data.
     */
    public DataStatistics getStatistics() {
        return new DataStatistics(
                database.getCount(),
                database.getUnusedCount());
    }

    /**
     * Statistics about collected data.
     */
    public static class DataStatistics {
        public int totalSamples;
        public int unusedSamples;

        public DataStatistics(int total, int unused) {
            this.totalSamples = total;
            this.unusedSamples = unused;
        }

        public int getUsedSamples() {
            return totalSamples - unusedSamples;
        }

        @Override
        public String toString() {
            return String.format("Total: %d, Unused: %d, Used: %d",
                    totalSamples, unusedSamples, getUsedSamples());
        }
    }
}
