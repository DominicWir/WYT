package com.example.wytv2.pdr.calibration;

/**
 * Utility class for calibrating step detection thresholds.
 * Uses gradient descent to find optimal threshold based on actual vs detected
 * steps.
 */
public class ThresholdCalibrator {

    // Learning rate for threshold adjustment (0.8 = adjust by 80% of error)
    private static final float LEARNING_RATE = 0.8f;

    // Minimum and maximum allowed thresholds
    private static final float MIN_THRESHOLD = 0.5f;
    private static final float MAX_THRESHOLD = 2.0f;

    // Acceptable error margin (10%)
    private static final float ACCEPTABLE_ERROR = 0.1f;

    /**
     * Calculate optimal threshold based on calibration data.
     * 
     * @param actualSteps      Number of steps user actually walked
     * @param detectedSteps    Number of steps detected by algorithm
     * @param currentThreshold Current detection threshold
     * @return Calibrated threshold value
     */
    public static float calculateOptimalThreshold(
            int actualSteps,
            int detectedSteps,
            float currentThreshold) {

        if (actualSteps <= 0) {
            throw new IllegalArgumentException("Actual steps must be positive");
        }

        // Calculate error rate
        float error = (detectedSteps - actualSteps) / (float) actualSteps;

        // If within acceptable margin, keep current threshold
        if (Math.abs(error) < ACCEPTABLE_ERROR) {
            return currentThreshold;
        }

        // Calculate new threshold
        // IMPORTANT: In this algorithm, LOWER threshold = MORE sensitive
        // (because binary[i] = value > threshold, so lower threshold means more values
        // pass)
        //
        // Too few steps detected → INCREASE threshold (make MORE sensitive)
        // Too many steps detected → DECREASE threshold (make LESS sensitive)
        //
        // This is OPPOSITE of intuition, so we NEGATE the error
        float newThreshold = currentThreshold * (1 - error * LEARNING_RATE);

        // Clamp to reasonable range
        return clampThreshold(newThreshold);
    }

    /**
     * Calculate accuracy percentage.
     * 
     * @param actualSteps   Actual number of steps
     * @param detectedSteps Detected number of steps
     * @return Accuracy as percentage (0-100)
     */
    public static float calculateAccuracy(int actualSteps, int detectedSteps) {
        if (actualSteps == 0)
            return 0f;

        float error = Math.abs(actualSteps - detectedSteps) / (float) actualSteps;
        return (1.0f - error) * 100f;
    }

    /**
     * Determine if calibration is acceptable.
     * 
     * @param actualSteps   Actual number of steps
     * @param detectedSteps Detected number of steps
     * @return true if accuracy is within acceptable range
     */
    public static boolean isCalibrationAcceptable(int actualSteps, int detectedSteps) {
        float error = Math.abs(actualSteps - detectedSteps) / (float) actualSteps;
        return error < ACCEPTABLE_ERROR;
    }

    /**
     * Get recommended action based on calibration results.
     * 
     * @param actualSteps   Actual number of steps
     * @param detectedSteps Detected number of steps
     * @return Human-readable recommendation
     */
    public static String getCalibrationRecommendation(int actualSteps, int detectedSteps) {
        float error = (detectedSteps - actualSteps) / (float) actualSteps;

        if (Math.abs(error) < ACCEPTABLE_ERROR) {
            return "Calibration successful! Threshold is optimal.";
        } else if (detectedSteps > actualSteps) {
            return "Too many steps detected. Threshold will be decreased (less sensitive).";
        } else {
            return "Too few steps detected. Threshold will be increased (more sensitive).";
        }
    }

    /**
     * Clamp threshold to valid range.
     */
    private static float clampThreshold(float threshold) {
        return Math.max(MIN_THRESHOLD, Math.min(MAX_THRESHOLD, threshold));
    }

    /**
     * Data class for calibration iteration results.
     */
    public static class CalibrationResult {
        public final int actualSteps;
        public final int detectedSteps;
        public final float oldThreshold;
        public final float newThreshold;
        public final float accuracy;
        public final String recommendation;

        public CalibrationResult(
                int actualSteps,
                int detectedSteps,
                float oldThreshold,
                float newThreshold) {
            this.actualSteps = actualSteps;
            this.detectedSteps = detectedSteps;
            this.oldThreshold = oldThreshold;
            this.newThreshold = newThreshold;
            this.accuracy = calculateAccuracy(actualSteps, detectedSteps);
            this.recommendation = getCalibrationRecommendation(actualSteps, detectedSteps);
        }

        @Override
        public String toString() {
            return String.format(
                    "CalibrationResult{actual=%d, detected=%d, threshold=%.2f→%.2f, accuracy=%.1f%%}",
                    actualSteps, detectedSteps, oldThreshold, newThreshold, accuracy);
        }
    }
}
