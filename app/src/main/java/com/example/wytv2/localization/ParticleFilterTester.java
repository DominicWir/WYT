package com.example.wytv2.localization;

import android.util.Log;
import java.util.ArrayList;
import java.util.List;

/**
 * Testing utility for Particle Filter accuracy measurement.
 * Tracks position estimates vs ground truth and calculates error metrics.
 */
public class ParticleFilterTester {
    private static final String TAG = "PFTester";

    // Ground truth path: (0,0) → (10,0) in 1m steps
    private static final float[][] GROUND_TRUTH_PATH = {
            { 0.0f, 0.0f },
            { 1.0f, 0.0f },
            { 2.0f, 0.0f },
            { 3.0f, 0.0f },
            { 4.0f, 0.0f },
            { 5.0f, 0.0f },
            { 6.0f, 0.0f },
            { 7.0f, 0.0f },
            { 8.0f, 0.0f },
            { 9.0f, 0.0f },
            { 10.0f, 0.0f }
    };

    private List<PositionError> errors;
    private int currentStepIndex;
    private boolean isTracking;
    private com.example.wytv2.visualization.PathTracker pathTracker;
    private com.example.wytv2.analysis.ParticleFilterDataCollector dataCollector;

    public static class PositionError {
        public float groundTruthX;
        public float groundTruthY;
        public float estimatedX;
        public float estimatedY;
        public float error;
        public float confidence;
        public long timestamp;

        public PositionError(float gtX, float gtY, float estX, float estY,
                float confidence, long timestamp) {
            this.groundTruthX = gtX;
            this.groundTruthY = gtY;
            this.estimatedX = estX;
            this.estimatedY = estY;
            this.confidence = confidence;
            this.timestamp = timestamp;

            // Calculate Euclidean distance error
            float dx = estX - gtX;
            float dy = estY - gtY;
            this.error = (float) Math.sqrt(dx * dx + dy * dy);
        }

        @Override
        public String toString() {
            return String.format("GT:(%.1f,%.1f) Est:(%.1f,%.1f) Error:%.2fm Conf:%.2f",
                    groundTruthX, groundTruthY, estimatedX, estimatedY, error, confidence);
        }
    }

    public ParticleFilterTester() {
        errors = new ArrayList<>();
        currentStepIndex = 0;
        isTracking = false;
        dataCollector = new com.example.wytv2.analysis.ParticleFilterDataCollector();
    }

    /**
     * Set path tracker for visualization.
     */
    public void setPathTracker(com.example.wytv2.visualization.PathTracker tracker) {
        this.pathTracker = tracker;
    }

    /**
     * Start tracking position accuracy.
     */
    public void startTracking() {
        errors.clear();
        currentStepIndex = 0;
        isTracking = true;

        // Start data collection
        if (dataCollector != null) {
            dataCollector.startTest();
        }

        Log.d(TAG, "Started accuracy tracking");
    }

    /**
     * Stop tracking and print results.
     */
    public void stopTracking() {
        isTracking = false;
        printResults();

        // Analyze and export data
        if (dataCollector != null) {
            dataCollector.analyzeData();
        }
    }

    /**
     * Record a position estimate after each step.
     * 
     * @param estimatedPos Position estimate from particle filter
     */
    public void recordStep(Position estimatedPos) {
        recordStep(estimatedPos, 0, 0); // Default values for spread and neff
    }

    /**
     * Record a position estimate with particle filter metrics.
     * 
     * @param estimatedPos   Position estimate from particle filter
     * @param particleSpread Current particle spread in meters
     * @param neff           Effective sample size
     */
    public void recordStep(Position estimatedPos, float particleSpread, float neff) {
        if (!isTracking || estimatedPos == null) {
            return;
        }

        if (currentStepIndex >= GROUND_TRUTH_PATH.length) {
            Log.w(TAG, "Exceeded ground truth path length");
            stopTracking();
            return;
        }

        float[] groundTruth = GROUND_TRUTH_PATH[currentStepIndex];

        PositionError error = new PositionError(
                groundTruth[0], groundTruth[1],
                estimatedPos.x, estimatedPos.y,
                estimatedPos.confidence,
                System.currentTimeMillis());

        errors.add(error);

        // Add to path tracker for visualization
        if (pathTracker != null) {
            pathTracker.addGroundTruthPosition(
                    groundTruth[0],
                    groundTruth[1],
                    error.timestamp);
        }

        // Add to data collector for analysis
        if (dataCollector != null) {
            dataCollector.addDataPoint(
                    groundTruth[0], groundTruth[1],
                    estimatedPos.x, estimatedPos.y,
                    estimatedPos.confidence,
                    particleSpread,
                    neff);
        }

        Log.d(TAG, String.format("Step %d: %s", currentStepIndex + 1, error.toString()));

        currentStepIndex++;

        // Auto-stop when path complete
        if (currentStepIndex >= GROUND_TRUTH_PATH.length) {
            stopTracking();
        }
    }

    /**
     * Print comprehensive accuracy results.
     */
    private void printResults() {
        if (errors.isEmpty()) {
            Log.w(TAG, "No data to analyze");
            return;
        }

        // Calculate statistics
        float sumError = 0;
        float maxError = 0;
        float minError = Float.MAX_VALUE;
        float sumConfidence = 0;

        for (PositionError e : errors) {
            sumError += e.error;
            maxError = Math.max(maxError, e.error);
            minError = Math.min(minError, e.error);
            sumConfidence += e.confidence;
        }

        float meanError = sumError / errors.size();
        float meanConfidence = sumConfidence / errors.size();

        // Calculate standard deviation
        float sumSquaredDiff = 0;
        for (PositionError e : errors) {
            float diff = e.error - meanError;
            sumSquaredDiff += diff * diff;
        }
        float stdDev = (float) Math.sqrt(sumSquaredDiff / errors.size());

        // Print results
        Log.d(TAG, "========================================");
        Log.d(TAG, "PARTICLE FILTER ACCURACY RESULTS");
        Log.d(TAG, "========================================");
        Log.d(TAG, String.format("Total steps: %d", errors.size()));
        Log.d(TAG, String.format("Mean error: %.2f m", meanError));
        Log.d(TAG, String.format("Std deviation: %.2f m", stdDev));
        Log.d(TAG, String.format("Min error: %.2f m", minError));
        Log.d(TAG, String.format("Max error: %.2f m", maxError));
        Log.d(TAG, String.format("Mean confidence: %.2f", meanConfidence));
        Log.d(TAG, "========================================");

        // Print detailed breakdown
        Log.d(TAG, "Detailed breakdown:");
        for (int i = 0; i < errors.size(); i++) {
            Log.d(TAG, String.format("  Step %d: %s", i + 1, errors.get(i).toString()));
        }

        // Accuracy assessment
        String assessment;
        if (meanError < 1.0f) {
            assessment = "EXCELLENT (< 1m)";
        } else if (meanError < 2.0f) {
            assessment = "GOOD (< 2m) - Target achieved!";
        } else if (meanError < 3.0f) {
            assessment = "ACCEPTABLE (< 3m)";
        } else {
            assessment = "NEEDS IMPROVEMENT (> 3m)";
        }

        Log.d(TAG, String.format("Assessment: %s", assessment));
        Log.d(TAG, "========================================");
    }

    /**
     * Get current tracking status.
     */
    public boolean isTracking() {
        return isTracking;
    }

    /**
     * Get number of steps recorded.
     */
    public int getStepCount() {
        return errors.size();
    }

    /**
     * Get mean error so far.
     */
    public float getCurrentMeanError() {
        if (errors.isEmpty())
            return 0;

        float sum = 0;
        for (PositionError e : errors) {
            sum += e.error;
        }
        return sum / errors.size();
    }
}
