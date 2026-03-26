package com.example.wytv2.ml;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

/**
 * Scheduler for triggering model retraining with adjustable frequency.
 * Supports multiple trigger types: step count, time-based, accuracy-based, and
 * manual.
 */
public class RetrainingScheduler {
    private static final String TAG = "RetrainingScheduler";
    private static final String PREFS_NAME = "retraining_prefs";
    private static final String KEY_LAST_TRAINING_TIME = "last_training_time";
    private static final String KEY_TRAINING_COUNT = "training_count";

    private Context context;
    private StepDataCollector dataCollector;

    // Adjustable parameters (prototype defaults)
    private int retrainingInterval = 50; // Every 50 steps for rapid prototyping
    private int minSamplesRequired = 30; // Minimum 30 samples needed
    private RetrainingTrigger trigger = RetrainingTrigger.STEP_COUNT;
    private long timeIntervalMs = 3600000; // 1 hour for time-based trigger

    // Tracking
    private int stepsSinceLastTraining = 0;
    private long lastTrainingTime = 0;

    public enum RetrainingTrigger {
        STEP_COUNT, // Trigger every N steps
        TIME_BASED, // Trigger every N hours
        DATA_BASED, // Trigger when N new samples collected
        ACCURACY_DROP, // Trigger when accuracy drops below threshold
        MANUAL // User-triggered only
    }

    public RetrainingScheduler(Context context, StepDataCollector dataCollector) {
        this.context = context;
        this.dataCollector = dataCollector;
        loadState();
    }

    /**
     * Check if retraining should be triggered and schedule if needed.
     * Call this after each step detection.
     */
    public void checkAndScheduleRetraining(int currentStepCount) {
        stepsSinceLastTraining++;

        boolean shouldRetrain = false;
        String reason = "";

        switch (trigger) {
            case STEP_COUNT:
                if (stepsSinceLastTraining >= retrainingInterval) {
                    shouldRetrain = true;
                    reason = "Step count threshold reached: " + stepsSinceLastTraining;
                }
                break;

            case TIME_BASED:
                long timeSinceLastTraining = System.currentTimeMillis() - lastTrainingTime;
                if (timeSinceLastTraining >= timeIntervalMs) {
                    shouldRetrain = true;
                    reason = "Time interval reached: " + (timeSinceLastTraining / 1000) + "s";
                }
                break;

            case DATA_BASED:
                int unusedSamples = dataCollector.getUnusedSamplesCount();
                if (unusedSamples >= retrainingInterval) {
                    shouldRetrain = true;
                    reason = "New data threshold reached: " + unusedSamples + " samples";
                }
                break;

            case ACCURACY_DROP:
                // TODO: Implement accuracy tracking
                // For now, fall back to step count
                if (stepsSinceLastTraining >= retrainingInterval * 2) {
                    shouldRetrain = true;
                    reason = "Fallback: step count threshold";
                }
                break;

            case MANUAL:
                // Only trigger manually
                shouldRetrain = false;
                break;
        }

        if (shouldRetrain) {
            scheduleRetraining(reason);
        }
    }

    /**
     * Schedule a retraining job.
     */
    private void scheduleRetraining(String reason) {
        // Check if enough data is available
        int availableData = dataCollector.getUnusedSamplesCount();
        if (availableData < minSamplesRequired) {
            Log.d(TAG, String.format(
                    "Not enough data for retraining: %d/%d samples",
                    availableData, minSamplesRequired));
            return;
        }

        Log.i(TAG, String.format(
                "Scheduling retraining: %s (available data: %d samples)",
                reason, availableData));

        // Perform retraining immediately (lightweight threshold optimization)
        ModelRetrainer retrainer = new ModelRetrainer(context, dataCollector);
        ModelRetrainer.RetrainingResult result = retrainer.retrain(500);

        Log.i(TAG, "Retraining result: " + result);

        // NOTE: Current model is rule-based (no actual model file to save)
        // Model persistence infrastructure is ready for future ML model integration
        // When we have a TFLite model, add modelFile field to RetrainingResult
        // and uncomment the following code to save it:
        /*
         * if (result.success && result.modelFile != null) {
         * boolean saved =
         * com.example.wytv2.SimplifiedActivityModel.saveModel(result.modelFile);
         * if (saved) {
         * Log.i(TAG, "Retrained model saved successfully");
         * com.example.wytv2.SimplifiedActivityModel.initialize(context);
         * }
         * }
         */

        // Reset counter and update timestamp
        stepsSinceLastTraining = 0;
        lastTrainingTime = System.currentTimeMillis();
        saveState();

        // Notify listener
        if (retrainingListener != null) {
            retrainingListener.onRetrainingScheduled(reason, availableData);
            retrainingListener.onRetrainingCompleted(result.success, result.message);
        }
    }

    /**
     * Manually trigger retraining (ignores automatic triggers).
     */
    public void triggerManualRetraining() {
        scheduleRetraining("Manual trigger");
    }

    // ========================================
    // Configuration Methods
    // ========================================

    /**
     * Set retraining interval (steps or samples depending on trigger type).
     */
    public void setRetrainingInterval(int interval) {
        this.retrainingInterval = interval;
        Log.d(TAG, "Retraining interval set to: " + interval);
    }

    public int getRetrainingInterval() {
        return retrainingInterval;
    }

    /**
     * Set minimum samples required before retraining.
     */
    public void setMinSamplesRequired(int samples) {
        this.minSamplesRequired = samples;
        Log.d(TAG, "Minimum samples required set to: " + samples);
    }

    public int getMinSamplesRequired() {
        return minSamplesRequired;
    }

    /**
     * Set trigger type.
     */
    public void setTrigger(RetrainingTrigger trigger) {
        this.trigger = trigger;
        Log.d(TAG, "Retraining trigger set to: " + trigger);
    }

    public RetrainingTrigger getTrigger() {
        return trigger;
    }

    /**
     * Set time interval for TIME_BASED trigger (in milliseconds).
     */
    public void setTimeInterval(long intervalMs) {
        this.timeIntervalMs = intervalMs;
        Log.d(TAG, "Time interval set to: " + (intervalMs / 1000) + "s");
    }

    public long getTimeInterval() {
        return timeIntervalMs;
    }

    // ========================================
    // State Persistence
    // ========================================

    private void loadState() {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        lastTrainingTime = prefs.getLong(KEY_LAST_TRAINING_TIME, 0);
        int trainingCount = prefs.getInt(KEY_TRAINING_COUNT, 0);
        Log.d(TAG, String.format("Loaded state: last training %d, count %d",
                lastTrainingTime, trainingCount));
    }

    private void saveState() {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        int trainingCount = prefs.getInt(KEY_TRAINING_COUNT, 0) + 1;
        prefs.edit()
                .putLong(KEY_LAST_TRAINING_TIME, lastTrainingTime)
                .putInt(KEY_TRAINING_COUNT, trainingCount)
                .apply();
        Log.d(TAG, "State saved: training count " + trainingCount);
    }

    /**
     * Get statistics about retraining.
     */
    public RetrainingStats getStats() {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        return new RetrainingStats(
                prefs.getLong(KEY_LAST_TRAINING_TIME, 0),
                prefs.getInt(KEY_TRAINING_COUNT, 0),
                stepsSinceLastTraining,
                dataCollector.getUnusedSamplesCount());
    }

    // ========================================
    // Listener Interface
    // ========================================

    private RetrainingListener retrainingListener;

    public interface RetrainingListener {
        void onRetrainingScheduled(String reason, int availableData);

        void onRetrainingCompleted(boolean success, String message);
    }

    public void setRetrainingListener(RetrainingListener listener) {
        this.retrainingListener = listener;
    }

    /**
     * Statistics about retraining.
     */
    public static class RetrainingStats {
        public long lastTrainingTime;
        public int totalTrainingCount;
        public int stepsSinceLastTraining;
        public int unusedSamples;

        public RetrainingStats(long lastTime, int count, int steps, int samples) {
            this.lastTrainingTime = lastTime;
            this.totalTrainingCount = count;
            this.stepsSinceLastTraining = steps;
            this.unusedSamples = samples;
        }

        public String getLastTrainingTimeString() {
            if (lastTrainingTime == 0) {
                return "Never";
            }
            long elapsed = System.currentTimeMillis() - lastTrainingTime;
            long hours = elapsed / 3600000;
            long minutes = (elapsed % 3600000) / 60000;
            return String.format("%dh %dm ago", hours, minutes);
        }

        @Override
        public String toString() {
            return String.format(
                    "Last: %s, Total: %d, Steps since: %d, Unused samples: %d",
                    getLastTrainingTimeString(), totalTrainingCount,
                    stepsSinceLastTraining, unusedSamples);
        }
    }
}
