package com.example.wytv2;

import android.content.Context;
import android.util.Log;

import com.example.wytv2.ml.TFLiteActivityModel;

import java.io.File;

/**
 * Activity recognition model wrapper.
 * 
 * Delegates to TFLiteActivityModel for inference when a .tflite model is
 * available,
 * falling back to rule-based classification otherwise.
 * 
 * Also manages model persistence (save/load retrained models) via
 * ModelFileManager.
 */
public class SimplifiedActivityModel {
    private static final String TAG = "SimplifiedActivityModel";

    private static com.example.wytv2.ml.ModelFileManager modelFileManager;
    private static TFLiteActivityModel tfliteModel;
    private static boolean isInitialized = false;

    private static final String[] ACTIVITIES = {
            "Stationary",
            "Walking",
            "Running",
            "Stairs Up",
            "Stairs Down"
    };

    /**
     * Initialize model with TFLite and persistence support.
     * Call this once during app startup.
     * 
     * @param context Application context
     */
    public static void initialize(Context context) {
        if (isInitialized) {
            Log.d(TAG, "Model already initialized");
            return;
        }

        modelFileManager = new com.example.wytv2.ml.ModelFileManager(context);

        // Initialize TFLite model
        tfliteModel = new TFLiteActivityModel();
        boolean modelLoaded = tfliteModel.initialize(context);

        // Log model status
        com.example.wytv2.ml.ModelFileManager.ModelVersionInfo version = modelFileManager.getVersionInfo();
        Log.d(TAG, String.format(
                "Model initialized: %s (version %d), TFLite: %s",
                version.type, version.version,
                modelLoaded ? "LOADED" : "FALLBACK (rule-based)"));

        isInitialized = true;
    }

    /**
     * Save updated model after retraining.
     * 
     * @param newModelFile The retrained model file
     * @return true if save successful, false otherwise
     */
    public static boolean saveModel(File newModelFile) {
        if (modelFileManager == null) {
            Log.e(TAG, "Model not initialized, cannot save");
            return false;
        }

        boolean success = modelFileManager.saveRetrainedModel(newModelFile);
        if (success) {
            modelFileManager.updateVersionInfo();
            Log.d(TAG, "Model saved and version updated");

            // Reload the TFLite model with the new weights
            if (tfliteModel != null) {
                tfliteModel.close();
                tfliteModel.initialize(null); // Will use ModelFileManager to find current model
            }
        }
        return success;
    }

    /**
     * Get current model version information.
     * 
     * @return ModelVersionInfo or null if not initialized
     */
    public static com.example.wytv2.ml.ModelFileManager.ModelVersionInfo getModelVersion() {
        if (modelFileManager == null) {
            return null;
        }
        return modelFileManager.getVersionInfo();
    }

    /**
     * Reset to base model (clear all retrained models).
     * Useful for testing or if user wants to start fresh.
     */
    public static void resetToBaseModel() {
        if (modelFileManager != null) {
            modelFileManager.resetToBaseModel();
            Log.d(TAG, "Reset to base model");
        }
    }

    /**
     * Check if TFLite model is loaded (vs rule-based fallback).
     */
    public static boolean isTFLiteLoaded() {
        return tfliteModel != null && tfliteModel.isModelLoaded();
    }

    /**
     * Predict activity from sensor features.
     * 
     * @param features 2D array of shape [50, 69] (sequence_length, feature_dim)
     * @return Predicted activity name
     */
    public static String predict(float[][] features) {
        if (tfliteModel != null) {
            return tfliteModel.predict(features);
        }
        // Safety net — should not reach here after initialization
        return "Unknown";
    }

    /**
     * Get prediction with confidence scores.
     */
    public static PredictionResult predictWithConfidence(float[][] features) {
        if (tfliteModel != null) {
            TFLiteActivityModel.PredictionResult result = tfliteModel.predictWithConfidence(features);
            return new PredictionResult(result.activity, result.activityIndex,
                    result.confidence, result.probabilities);
        }
        return new PredictionResult("Unknown", 0, 0.0f, new float[5]);
    }

    /**
     * Release model resources. Call on app shutdown.
     */
    public static void close() {
        if (tfliteModel != null) {
            tfliteModel.close();
            tfliteModel = null;
        }
        isInitialized = false;
    }

    /**
     * Result class containing prediction and confidence.
     */
    public static class PredictionResult {
        public final String activity;
        public final int activityIndex;
        public final float confidence;
        public final float[] probabilities;

        public PredictionResult(String activity, int activityIndex,
                float confidence, float[] probabilities) {
            this.activity = activity;
            this.activityIndex = activityIndex;
            this.confidence = confidence;
            this.probabilities = probabilities;
        }

        @Override
        public String toString() {
            return String.format("%s (%.1f%%)", activity, confidence * 100);
        }
    }
}
