package com.example.wytv2.ml;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import java.util.List;

/**
 * Performs "retraining" by optimizing thresholds based on collected step data.
 * This is a lightweight alternative to full neural network retraining.
 */
public class ModelRetrainer {
    private static final String TAG = "ModelRetrainer";
    private static final String PREFS_NAME = "model_retrainer_prefs";
    private static final String KEY_CURRENT_THRESHOLD = "current_threshold";
    private static final String KEY_CURRENT_ACCURACY = "current_accuracy";
    private static final String KEY_BEST_THRESHOLD = "best_threshold";
    private static final String KEY_BEST_ACCURACY = "best_accuracy";

    private Context context;
    private StepDataCollector dataCollector;
    private SharedPreferences prefs;

    // Callbacks
    private ThresholdUpdateListener thresholdUpdateListener;

    public interface ThresholdUpdateListener {
        void onThresholdUpdated(float newThreshold, String reason);
    }

    public ModelRetrainer(Context context, StepDataCollector dataCollector) {
        this.context = context;
        this.dataCollector = dataCollector;
        this.prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);

        // Load saved state
        float savedThreshold = prefs.getFloat(KEY_CURRENT_THRESHOLD, 1.2f);
        Log.d(TAG, "Loaded saved threshold: " + savedThreshold);
    }

    /**
     * Perform retraining by analyzing collected data and optimizing threshold.
     * 
     * @param maxSamples Maximum number of samples to use (use recent data)
     * @return Result of retraining operation
     */
    public RetrainingResult retrain(int maxSamples) {
        Log.i(TAG, "Starting retraining with max " + maxSamples + " samples");

        // Get unused training data
        List<StepDataPoint> data = dataCollector.getUnusedTrainingData(maxSamples);

        if (data.size() < 30) {
            String msg = "Insufficient data: " + data.size() + "/30 samples";
            Log.w(TAG, msg);
            return new RetrainingResult(false, getCurrentAccuracy(),
                    getCurrentThreshold(), msg);
        }

        Log.d(TAG, "Analyzing " + data.size() + " samples");

        // Analyze data to find optimal threshold
        StepDataAnalyzer.AnalysisResult analysis = StepDataAnalyzer.analyzeStepData(data);

        if (analysis == null) {
            String msg = "Analysis failed";
            Log.e(TAG, msg);
            return new RetrainingResult(false, getCurrentAccuracy(),
                    getCurrentThreshold(), msg);
        }

        Log.i(TAG, "Analysis result: " + analysis);

        // Get current performance
        float currentAccuracy = getCurrentAccuracy();
        float currentThreshold = getCurrentThreshold();

        // Check if new threshold improves accuracy
        float improvementThreshold = 0.02f; // Require 2% improvement

        if (analysis.accuracy > currentAccuracy + improvementThreshold) {
            // Apply new threshold
            applyNewThreshold(analysis.optimalThreshold, analysis.accuracy);

            // Mark data as used
            dataCollector.markDataAsUsed(data);

            String msg = String.format(
                    "Improved from %.1f%% to %.1f%% (threshold: %.2f → %.2f)",
                    currentAccuracy * 100, analysis.accuracy * 100,
                    currentThreshold, analysis.optimalThreshold);

            Log.i(TAG, "✓ " + msg);

            // Notify listener
            if (thresholdUpdateListener != null) {
                thresholdUpdateListener.onThresholdUpdated(
                        analysis.optimalThreshold, msg);
            }

            return new RetrainingResult(true, analysis.accuracy,
                    analysis.optimalThreshold, msg);

        } else {
            String msg = String.format(
                    "No improvement: current %.1f%%, new %.1f%%",
                    currentAccuracy * 100, analysis.accuracy * 100);

            Log.d(TAG, msg);

            // Still mark data as used to avoid reprocessing
            dataCollector.markDataAsUsed(data);

            return new RetrainingResult(false, currentAccuracy,
                    currentThreshold, msg);
        }
    }

    /**
     * Apply new threshold and save to preferences.
     */
    private void applyNewThreshold(float threshold, float accuracy) {
        prefs.edit()
                .putFloat(KEY_CURRENT_THRESHOLD, threshold)
                .putFloat(KEY_CURRENT_ACCURACY, accuracy)
                .apply();

        // Update best threshold if this is better
        float bestAccuracy = prefs.getFloat(KEY_BEST_ACCURACY, 0.0f);
        if (accuracy > bestAccuracy) {
            prefs.edit()
                    .putFloat(KEY_BEST_THRESHOLD, threshold)
                    .putFloat(KEY_BEST_ACCURACY, accuracy)
                    .apply();
            Log.i(TAG, "New best threshold: " + threshold);
        }
    }

    /**
     * Get current threshold.
     */
    public float getCurrentThreshold() {
        return prefs.getFloat(KEY_CURRENT_THRESHOLD, 1.2f);
    }

    /**
     * Get current accuracy.
     */
    public float getCurrentAccuracy() {
        return prefs.getFloat(KEY_CURRENT_ACCURACY, 0.5f);
    }

    /**
     * Get best threshold ever achieved.
     */
    public float getBestThreshold() {
        return prefs.getFloat(KEY_BEST_THRESHOLD, 1.2f);
    }

    /**
     * Get best accuracy ever achieved.
     */
    public float getBestAccuracy() {
        return prefs.getFloat(KEY_BEST_ACCURACY, 0.5f);
    }

    /**
     * Set threshold update listener.
     */
    public void setThresholdUpdateListener(ThresholdUpdateListener listener) {
        this.thresholdUpdateListener = listener;
    }

    /**
     * Get retraining statistics.
     */
    public RetrainingStats getStats() {
        return new RetrainingStats(
                getCurrentThreshold(),
                getCurrentAccuracy(),
                getBestThreshold(),
                getBestAccuracy());
    }

    /**
     * Result of a retraining operation.
     */
    public static class RetrainingResult {
        public boolean success;
        public float accuracy;
        public float threshold;
        public String message;

        public RetrainingResult(boolean success, float accuracy,
                float threshold, String message) {
            this.success = success;
            this.accuracy = accuracy;
            this.threshold = threshold;
            this.message = message;
        }

        @Override
        public String toString() {
            return String.format("%s: %s (threshold: %.2f, accuracy: %.1f%%)",
                    success ? "SUCCESS" : "NO CHANGE",
                    message, threshold, accuracy * 100);
        }
    }

    /**
     * Statistics about model retraining.
     */
    public static class RetrainingStats {
        public float currentThreshold;
        public float currentAccuracy;
        public float bestThreshold;
        public float bestAccuracy;

        public RetrainingStats(float currentThreshold, float currentAccuracy,
                float bestThreshold, float bestAccuracy) {
            this.currentThreshold = currentThreshold;
            this.currentAccuracy = currentAccuracy;
            this.bestThreshold = bestThreshold;
            this.bestAccuracy = bestAccuracy;
        }

        @Override
        public String toString() {
            return String.format(
                    "Current: %.2f (%.1f%%), Best: %.2f (%.1f%%)",
                    currentThreshold, currentAccuracy * 100,
                    bestThreshold, bestAccuracy * 100);
        }
    }
}
