package com.example.wytv2.ml;

import android.util.Log;

import java.util.ArrayList;
import java.util.List;

/**
 * Analyzes collected step data to find optimal detection thresholds.
 * Uses statistical methods and ROC curve analysis.
 */
public class StepDataAnalyzer {
    private static final String TAG = "StepDataAnalyzer";

    /**
     * Analyze step data and find optimal threshold.
     */
    public static AnalysisResult analyzeStepData(List<StepDataPoint> data) {
        if (data == null || data.size() < 10) {
            Log.w(TAG, "Insufficient data for analysis: " + (data != null ? data.size() : 0));
            return null;
        }

        // Separate true steps from false positives
        List<StepDataPoint> trueSteps = new ArrayList<>();
        List<StepDataPoint> falsePositives = new ArrayList<>();

        for (StepDataPoint point : data) {
            if (point.isActualStep) {
                trueSteps.add(point);
            } else {
                falsePositives.add(point);
            }
        }

        Log.d(TAG, String.format("Analyzing %d true steps, %d false positives",
                trueSteps.size(), falsePositives.size()));

        // Calculate feature statistics
        float avgTrueStepMagnitude = calculateAvgFeatureMagnitude(trueSteps);
        float avgFalsePositiveMagnitude = calculateAvgFeatureMagnitude(falsePositives);

        Log.d(TAG, String.format("Avg magnitudes - True: %.3f, False: %.3f",
                avgTrueStepMagnitude, avgFalsePositiveMagnitude));

        // Find optimal threshold using ROC analysis
        float optimalThreshold = findOptimalThreshold(trueSteps, falsePositives);

        // Calculate accuracy with optimal threshold
        float accuracy = calculateAccuracy(data, optimalThreshold);

        return new AnalysisResult(
                optimalThreshold,
                accuracy,
                trueSteps.size(),
                falsePositives.size(),
                avgTrueStepMagnitude,
                avgFalsePositiveMagnitude);
    }

    /**
     * Find optimal threshold using ROC curve analysis.
     * Maximizes F1 score (harmonic mean of precision and recall).
     */
    private static float findOptimalThreshold(
            List<StepDataPoint> trueSteps,
            List<StepDataPoint> falsePositives) {

        float bestThreshold = 1.2f;
        float bestF1Score = 0.0f;

        // Try different thresholds from 0.5 to 2.5
        for (float threshold = 0.5f; threshold <= 2.5f; threshold += 0.05f) {
            // Count true positives (true steps detected)
            int tp = countStepsDetectedAtThreshold(trueSteps, threshold);

            // Count false positives (false alarms detected)
            int fp = countStepsDetectedAtThreshold(falsePositives, threshold);

            // Count false negatives (true steps missed)
            int fn = trueSteps.size() - tp;

            // Avoid division by zero
            if (tp == 0)
                continue;

            // Calculate precision and recall
            float precision = tp / (float) (tp + fp + 0.001f);
            float recall = tp / (float) (tp + fn + 0.001f);

            // Calculate F1 score (harmonic mean)
            float f1 = 2 * (precision * recall) / (precision + recall + 0.001f);

            if (f1 > bestF1Score) {
                bestF1Score = f1;
                bestThreshold = threshold;

                Log.d(TAG, String.format(
                        "New best threshold: %.2f (F1: %.3f, P: %.3f, R: %.3f)",
                        threshold, f1, precision, recall));
            }
        }

        Log.i(TAG, String.format("Optimal threshold: %.2f (F1: %.3f)",
                bestThreshold, bestF1Score));

        return bestThreshold;
    }

    /**
     * Count how many steps would be detected at a given threshold.
     * Uses the stored threshold from when the step was detected.
     */
    private static int countStepsDetectedAtThreshold(
            List<StepDataPoint> steps, float threshold) {
        int count = 0;
        for (StepDataPoint step : steps) {
            // If the step was detected with a threshold <= our test threshold,
            // it would still be detected
            if (step.threshold <= threshold) {
                count++;
            }
        }
        return count;
    }

    /**
     * Calculate average feature magnitude for a set of steps.
     */
    private static float calculateAvgFeatureMagnitude(List<StepDataPoint> steps) {
        if (steps.isEmpty())
            return 0.0f;

        float sum = 0.0f;
        for (StepDataPoint step : steps) {
            // Calculate magnitude from accelerometer data
            float mag = (float) Math.sqrt(
                    step.accelerometer[0] * step.accelerometer[0] +
                            step.accelerometer[1] * step.accelerometer[1] +
                            step.accelerometer[2] * step.accelerometer[2]);
            sum += mag;
        }
        return sum / steps.size();
    }

    /**
     * Calculate accuracy with a given threshold.
     */
    private static float calculateAccuracy(List<StepDataPoint> data, float threshold) {
        int correct = 0;

        for (StepDataPoint point : data) {
            boolean wouldDetect = point.threshold <= threshold;
            boolean shouldDetect = point.isActualStep;

            if (wouldDetect == shouldDetect) {
                correct++;
            }
        }

        return correct / (float) data.size();
    }

    /**
     * Result of step data analysis.
     */
    public static class AnalysisResult {
        public float optimalThreshold;
        public float accuracy;
        public int trueStepCount;
        public int falsePositiveCount;
        public float avgTrueStepMagnitude;
        public float avgFalsePositiveMagnitude;

        public AnalysisResult(float threshold, float accuracy,
                int trueSteps, int falsePositives,
                float avgTrue, float avgFalse) {
            this.optimalThreshold = threshold;
            this.accuracy = accuracy;
            this.trueStepCount = trueSteps;
            this.falsePositiveCount = falsePositives;
            this.avgTrueStepMagnitude = avgTrue;
            this.avgFalsePositiveMagnitude = avgFalse;
        }

        @Override
        public String toString() {
            return String.format(
                    "Threshold: %.2f, Accuracy: %.1f%%, True: %d, False: %d",
                    optimalThreshold, accuracy * 100,
                    trueStepCount, falsePositiveCount);
        }
    }
}
