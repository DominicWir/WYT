package com.example.wytv2.pdr.algorithm;

import java.util.List;

public class BinarizationStepDetector {
    private int S = 18; // Sliding window size (half of a_sum)
    private float binaryThreshold = 9.80f;
    private int stepCount = 0;

    // Algorithm 1: Binarization-based step detection
    public boolean detectStep(List<Float> accelerationWindow) {
        if (accelerationWindow.size() < S) {
            return false;
        }

        // Phase 1: Acceleration binarization
        boolean[] binaryCodes = new boolean[S];
        for (int i = 0; i < S; i++) {
            binaryCodes[i] = accelerationWindow.get(i) > binaryThreshold;
        }

        // Phase 2: False peak detection and step confirmation
        int a = 0; // Counter for binary code "1"
        boolean stepDetected = false;

        for (int i = 0; i < S; i++) {
            if (binaryCodes[i]) {
                a++;

                // Check for false peak (a < S/2 followed by "0")
                if (a < (S / 2)) {
                    // Look ahead for zeros
                    boolean hasZerosAhead = false;
                    for (int j = i + 1; j < Math.min(i + 5, S); j++) {
                        if (!binaryCodes[j]) {
                            hasZerosAhead = true;
                            break;
                        }
                    }
                    if (hasZerosAhead) {
                        // False peak detected, restart detection
                        return false;
                    }
                }
            } else {
                if (a > 0) {
                    a++; // Count zeros too when looking for step completion
                }
            }

            // Check if step is complete (a > S)
            if (a > S) {
                stepDetected = true;
                break;
            }
        }

        if (stepDetected) {
            stepCount++;
            return true;
        }

        return false;
    }

    // Algorithm 2: Iterative method to find binary threshold
    public float findBinaryThreshold(List<Float> accelerationData, int trueStepCount) {
        float threshold = 9.80f;
        float increment = 0.1f;
        boolean flag = true;

        // Find maximum acceleration value
        float maxAcc = 0;
        for (float acc : accelerationData) {
            if (acc > maxAcc) maxAcc = acc;
        }

        while (flag && threshold < maxAcc) {
            threshold += increment;

            // Calculate step number using current threshold
            int detectedSteps = calculateStepCount(accelerationData, threshold);

            // Check if within tolerance (abs(N - n) ≤ 1)
            if (Math.abs(trueStepCount - detectedSteps) <= 1) {
                flag = false; // Found optimal threshold
            }
        }

        return threshold;
    }

    private int calculateStepCount(List<Float> data, float threshold) {
        int count = 0;
        int a = 0;
        boolean[] binaryWindow = new boolean[S];
        int windowIndex = 0;

        for (float acc : data) {
            // Fill sliding window
            binaryWindow[windowIndex % S] = acc > threshold;
            windowIndex++;

            if (windowIndex >= S) {
                // Check if window contains a step
                if (detectStepInWindow(binaryWindow)) {
                    count++;
                    // Reset for next step
                    binaryWindow = new boolean[S];
                    windowIndex = 0;
                }
            }
        }

        return count;
    }

    private boolean detectStepInWindow(boolean[] window) {
        int a = 0;
        for (int i = 0; i < S; i++) {
            if (window[i]) {
                a++;
            } else if (a > 0) {
                a++;
            }

            if (a > S) {
                return true;
            }
        }
        return false;
    }

    // Getters and setters
    public int getStepCount() { return stepCount; }
    public void resetStepCount() { stepCount = 0; }
    public float getThreshold() { return binaryThreshold; }
    public void setThreshold(float threshold) { this.binaryThreshold = threshold; }
    public int getWindowSize() { return S; }
    public void setWindowSize(int size) { this.S = size; }
}
