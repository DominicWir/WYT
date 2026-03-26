package com.example.wytv2.analysis;

import android.util.Log;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Collects and analyzes particle filter test data.
 * Exports results to CSV for detailed analysis.
 */
public class ParticleFilterDataCollector {
    private static final String TAG = "PFDataCollector";

    private List<TestDataPoint> dataPoints;
    private long testStartTime;
    private String testId;

    public static class TestDataPoint {
        public long timestamp;
        public float groundTruthX;
        public float groundTruthY;
        public float estimatedX;
        public float estimatedY;
        public float error;
        public float confidence;
        public float particleSpread;
        public float neff;
        public int stepNumber;

        public TestDataPoint(long timestamp, float gtX, float gtY, float estX, float estY,
                float confidence, float particleSpread, float neff, int stepNumber) {
            this.timestamp = timestamp;
            this.groundTruthX = gtX;
            this.groundTruthY = gtY;
            this.estimatedX = estX;
            this.estimatedY = estY;
            this.confidence = confidence;
            this.particleSpread = particleSpread;
            this.neff = neff;
            this.stepNumber = stepNumber;

            // Calculate error
            float dx = estX - gtX;
            float dy = estY - gtY;
            this.error = (float) Math.sqrt(dx * dx + dy * dy);
        }

        public String toCSV() {
            return String.format(Locale.US, "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
                    timestamp, stepNumber, groundTruthX, groundTruthY, estimatedX, estimatedY,
                    error, confidence, particleSpread, neff);
        }
    }

    public ParticleFilterDataCollector() {
        dataPoints = new ArrayList<>();
        testId = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
    }

    /**
     * Start a new test session.
     */
    public void startTest() {
        dataPoints.clear();
        testStartTime = System.currentTimeMillis();
        testId = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        Log.d(TAG, "Started test session: " + testId);
    }

    /**
     * Add a data point.
     */
    public void addDataPoint(float gtX, float gtY, float estX, float estY,
            float confidence, float particleSpread, float neff) {
        TestDataPoint point = new TestDataPoint(
                System.currentTimeMillis(),
                gtX, gtY, estX, estY,
                confidence, particleSpread, neff,
                dataPoints.size() + 1);
        dataPoints.add(point);
    }

    /**
     * Analyze collected data and print comprehensive report.
     */
    public AnalysisResults analyzeData() {
        if (dataPoints.isEmpty()) {
            Log.w(TAG, "No data to analyze");
            return null;
        }

        AnalysisResults results = new AnalysisResults();

        // Basic statistics
        float sumError = 0;
        float sumConfidence = 0;
        float sumSpread = 0;
        float sumNeff = 0;
        results.minError = Float.MAX_VALUE;
        results.maxError = 0;

        for (TestDataPoint point : dataPoints) {
            sumError += point.error;
            sumConfidence += point.confidence;
            sumSpread += point.particleSpread;
            sumNeff += point.neff;
            results.minError = Math.min(results.minError, point.error);
            results.maxError = Math.max(results.maxError, point.error);
        }

        results.totalSteps = dataPoints.size();
        results.meanError = sumError / results.totalSteps;
        results.meanConfidence = sumConfidence / results.totalSteps;
        results.meanParticleSpread = sumSpread / results.totalSteps;
        results.meanNeff = sumNeff / results.totalSteps;

        // Standard deviation
        float sumSquaredDiff = 0;
        for (TestDataPoint point : dataPoints) {
            float diff = point.error - results.meanError;
            sumSquaredDiff += diff * diff;
        }
        results.stdDevError = (float) Math.sqrt(sumSquaredDiff / results.totalSteps);

        // Error progression (first half vs second half)
        int halfPoint = results.totalSteps / 2;
        float firstHalfError = 0;
        float secondHalfError = 0;

        for (int i = 0; i < halfPoint; i++) {
            firstHalfError += dataPoints.get(i).error;
        }
        for (int i = halfPoint; i < results.totalSteps; i++) {
            secondHalfError += dataPoints.get(i).error;
        }

        results.firstHalfMeanError = firstHalfError / halfPoint;
        results.secondHalfMeanError = secondHalfError / (results.totalSteps - halfPoint);
        results.errorDrift = results.secondHalfMeanError - results.firstHalfMeanError;

        // Test duration
        if (!dataPoints.isEmpty()) {
            long duration = dataPoints.get(dataPoints.size() - 1).timestamp -
                    dataPoints.get(0).timestamp;
            results.testDurationSeconds = duration / 1000.0f;
        }

        // Print report
        printAnalysisReport(results);

        return results;
    }

    /**
     * Print detailed analysis report.
     */
    private void printAnalysisReport(AnalysisResults results) {
        Log.d(TAG, "========================================");
        Log.d(TAG, "PARTICLE FILTER DATA ANALYSIS");
        Log.d(TAG, "Test ID: " + testId);
        Log.d(TAG, "========================================");
        Log.d(TAG, "");
        Log.d(TAG, "ACCURACY METRICS:");
        Log.d(TAG, String.format("  Total steps: %d", results.totalSteps));
        Log.d(TAG, String.format("  Mean error: %.2f m", results.meanError));
        Log.d(TAG, String.format("  Std deviation: %.2f m", results.stdDevError));
        Log.d(TAG, String.format("  Min error: %.2f m", results.minError));
        Log.d(TAG, String.format("  Max error: %.2f m", results.maxError));
        Log.d(TAG, "");
        Log.d(TAG, "ERROR PROGRESSION:");
        Log.d(TAG, String.format("  First half mean: %.2f m", results.firstHalfMeanError));
        Log.d(TAG, String.format("  Second half mean: %.2f m", results.secondHalfMeanError));
        Log.d(TAG, String.format("  Drift: %.2f m (%s)",
                Math.abs(results.errorDrift),
                results.errorDrift > 0 ? "increasing" : "decreasing"));
        Log.d(TAG, "");
        Log.d(TAG, "PARTICLE FILTER METRICS:");
        Log.d(TAG, String.format("  Mean confidence: %.2f", results.meanConfidence));
        Log.d(TAG, String.format("  Mean particle spread: %.2f m", results.meanParticleSpread));
        Log.d(TAG, String.format("  Mean Neff: %.0f", results.meanNeff));
        Log.d(TAG, "");
        Log.d(TAG, "TEST DETAILS:");
        Log.d(TAG, String.format("  Duration: %.1f seconds", results.testDurationSeconds));
        Log.d(TAG, String.format("  Steps per second: %.2f",
                results.totalSteps / results.testDurationSeconds));
        Log.d(TAG, "");

        // Assessment
        String assessment;
        if (results.meanError < 1.0f) {
            assessment = "EXCELLENT (< 1m) ⭐⭐⭐";
        } else if (results.meanError < 2.0f) {
            assessment = "GOOD (< 2m) ⭐⭐ - Target achieved!";
        } else if (results.meanError < 3.0f) {
            assessment = "ACCEPTABLE (< 3m) ⭐";
        } else {
            assessment = "NEEDS IMPROVEMENT (> 3m) ❌";
        }

        Log.d(TAG, "ASSESSMENT: " + assessment);
        Log.d(TAG, "========================================");
    }

    /**
     * Export data to CSV file.
     */
    public boolean exportToCSV(File outputDir) {
        if (dataPoints.isEmpty()) {
            Log.w(TAG, "No data to export");
            return false;
        }

        try {
            File csvFile = new File(outputDir, "pf_test_" + testId + ".csv");
            FileWriter writer = new FileWriter(csvFile);

            // Write header
            writer.write("timestamp,step,gt_x,gt_y,est_x,est_y,error,confidence,particle_spread,neff\n");

            // Write data points
            for (TestDataPoint point : dataPoints) {
                writer.write(point.toCSV() + "\n");
            }

            writer.close();
            Log.d(TAG, "Data exported to: " + csvFile.getAbsolutePath());
            return true;

        } catch (IOException e) {
            Log.e(TAG, "Failed to export CSV", e);
            return false;
        }
    }

    /**
     * Results container.
     */
    public static class AnalysisResults {
        public int totalSteps;
        public float meanError;
        public float stdDevError;
        public float minError;
        public float maxError;
        public float meanConfidence;
        public float meanParticleSpread;
        public float meanNeff;
        public float firstHalfMeanError;
        public float secondHalfMeanError;
        public float errorDrift;
        public float testDurationSeconds;
    }
}
