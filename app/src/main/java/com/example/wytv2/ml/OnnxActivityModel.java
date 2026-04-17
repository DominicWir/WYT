package com.example.wytv2.ml;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.util.Collections;

/**
 * Activity classes:
 * 0: Stationary
 * 1: Walking
 * 2: Running
 * 3: Stairs Up
 * 4: Stairs Down
 * 
 * Falls back to rule-based classification if no .onnx model is available.
 */
public class OnnxActivityModel {
    private static final String TAG = "OnnxActivityModel";

    private static final String MODEL_ASSET_NAME = "activity_model.onnx";
    private static final String NORMALIZER_ASSET_NAME = "activity_normalizer.json";

    private static final int WINDOW_SIZE = 50;
    private static final int FEATURE_DIM = 69;
    private static final int NUM_CLASSES = 5;

    private static final String[] ACTIVITY_LABELS = {
            "Stationary", "Walking", "Running", "Stairs Up", "Stairs Down"
    };

    private OrtEnvironment ortEnv;
    private OrtSession ortSession;
    private boolean isModelLoaded = false;
    private Context context;

    // Normalizer parameters
    private float[] imuMean;
    private float[] imuScale;
    private float[] magMean;
    private float[] magScale;
    private boolean normalizerLoaded = false;

    /**
     * Copies model files from assets to internal storage so ORT can find
     * both the .onnx and companion .onnx.data file.
     */
    public boolean initialize(Context context) {
        this.context = context;

        loadNormalizer();

        try {
            ortEnv = OrtEnvironment.getEnvironment();

            // Try retrained model first
            ModelFileManager fileManager = new ModelFileManager(context);
            File currentModel = fileManager.getCurrentModelPath();

            if (currentModel != null && currentModel.exists() && currentModel.length() > 0) {
                ortSession = ortEnv.createSession(currentModel.getAbsolutePath());
                isModelLoaded = true;
                Log.i(TAG, "✓ Loaded retrained model: " + currentModel.getAbsolutePath());
            } else {
                File modelDir = new File(context.getFilesDir(), "onnx_model");
                if (!modelDir.exists())
                    modelDir.mkdirs();

                File modelFile = new File(modelDir, MODEL_ASSET_NAME);
                File dataFile = new File(modelDir, MODEL_ASSET_NAME + ".data");

                // Copy main model file
                copyAssetToFile(MODEL_ASSET_NAME, modelFile);

                // Copy companion data file if it exists
                try {
                    copyAssetToFile(MODEL_ASSET_NAME + ".data", dataFile);
                    Log.d(TAG, "Copied companion .data file");
                } catch (IOException e) {
                    Log.d(TAG, "No companion .data file (model is self-contained)");
                }

                if (modelFile.exists() && modelFile.length() > 0) {
                    ortSession = ortEnv.createSession(modelFile.getAbsolutePath());
                    isModelLoaded = true;
                    Log.i(TAG, "✓ Loaded base model from assets (" +
                            modelFile.length() / 1024 + " KB)");
                } else {
                    Log.w(TAG, "⚠ No ONNX model found — using rule-based fallback");
                    isModelLoaded = false;
                }
            }

            if (isModelLoaded)
                logModelDetails();

        } catch (Exception e) {
            Log.e(TAG, "Failed to load ONNX model: " + e.getMessage(), e);
            isModelLoaded = false;
        }

        return isModelLoaded;
    }

    /**
     * Run inference on a feature window.
     */
    public String predict(float[][] features) {
        return predictWithConfidence(features).activity;
    }

    /**
     * Run inference with confidence scores.
     */
    public PredictionResult predictWithConfidence(float[][] features) {
        if (features == null || features.length != WINDOW_SIZE || features[0].length != FEATURE_DIM) {
            return new PredictionResult("Unknown", 0, 0.0f, new float[NUM_CLASSES]);
        }

        if (!isModelLoaded) {
            return runRuleBasedFallback(features);
        }

        try {
            // Normalize
            float[][] normalized = normalizeFeatures(features);

            // Flatten to 1D for OnnxTensor: [1, 50, 69]
            float[] flatInput = new float[WINDOW_SIZE * FEATURE_DIM];
            for (int i = 0; i < WINDOW_SIZE; i++) {
                System.arraycopy(normalized[i], 0, flatInput, i * FEATURE_DIM, FEATURE_DIM);
            }

            // Create ONNX tensor
            long[] shape = { 1, WINDOW_SIZE, FEATURE_DIM };
            OnnxTensor inputTensor = OnnxTensor.createTensor(
                    ortEnv, FloatBuffer.wrap(flatInput), shape);

            // Run inference
            OrtSession.Result result = ortSession.run(
                    Collections.singletonMap("input", inputTensor));

            // Get output
            float[][] output = (float[][]) result.get(0).getValue();
            float[] logits = output[0];

            // Apply softmax
            float[] probabilities = softmax(logits);

            // Find best class
            int bestClass = 0;
            float bestProb = probabilities[0];
            for (int i = 1; i < NUM_CLASSES; i++) {
                if (probabilities[i] > bestProb) {
                    bestProb = probabilities[i];
                    bestClass = i;
                }
            }

            String activity = ACTIVITY_LABELS[bestClass];
            Log.d(TAG, String.format("Prediction: %s (%.1f%%) [%.2f, %.2f, %.2f, %.2f, %.2f]",
                    activity, bestProb * 100,
                    probabilities[0], probabilities[1], probabilities[2],
                    probabilities[3], probabilities[4]));

            // Clean up
            inputTensor.close();
            result.close();

            return new PredictionResult(activity, bestClass, bestProb, probabilities);

        } catch (Exception e) {
            Log.e(TAG, "ONNX inference failed: " + e.getMessage());
            return runRuleBasedFallback(features);
        }
    }

    private float[][] normalizeFeatures(float[][] features) {
        if (!normalizerLoaded)
            return features;

        float[][] normalized = new float[WINDOW_SIZE][FEATURE_DIM];
        for (int t = 0; t < WINDOW_SIZE; t++) {
            for (int f = 0; f < FEATURE_DIM; f++) {
                if (f < 6) {
                    // Accel (0-2) + Gyro (3-5)
                    normalized[t][f] = (features[t][f] - imuMean[f]) / imuScale[f];
                } else if (f < 9) {
                    // Magnetometer (6-8)
                    normalized[t][f] = (features[t][f] - magMean[f - 6]) / magScale[f - 6];
                } else {
                    normalized[t][f] = features[t][f];
                }
            }
        }
        return normalized;
    }

    private float[] softmax(float[] logits) {
        float[] probs = new float[logits.length];
        float max = Float.NEGATIVE_INFINITY;
        for (float l : logits)
            if (l > max)
                max = l;
        float sum = 0;
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++)
            probs[i] /= sum;
        return probs;
    }

    private PredictionResult runRuleBasedFallback(float[][] features) {
        float motionIntensity = 0, verticalSum = 0;
        for (float[] row : features) {
            motionIntensity += (float) Math.sqrt(row[0] * row[0] + row[1] * row[1] + row[2] * row[2]);
            verticalSum += row[2];
        }
        motionIntensity /= features.length;
        float verticalMean = verticalSum / features.length;

        float verticalVariance = 0;
        for (float[] row : features) {
            float diff = row[2] - verticalMean;
            verticalVariance += diff * diff;
        }
        verticalVariance = (float) Math.sqrt(verticalVariance / features.length);

        float sum = 0, count = 0;
        for (float[] row : features)
            for (float v : row) {
                sum += v;
                count++;
            }
        float mean = sum / count;
        float stdSum = 0;
        for (float[] row : features)
            for (float v : row) {
                float d = v - mean;
                stdSum += d * d;
            }
        float std = (float) Math.sqrt(stdSum / count);

        String activity;
        int idx;
        if (motionIntensity > 0.5f && std > 0.3f && std < 1.5f) {
            activity = "Walking";
            idx = 1;
        } else {
            activity = "Stationary";
            idx = 0;
        }

        float confidence = Math.min(0.95f, 0.6f + std * 0.2f);
        float[] probs = new float[NUM_CLASSES];
        probs[idx] = confidence;

        Log.d(TAG, "Rule-based fallback: " + activity);
        return new PredictionResult(activity, idx, confidence, probs);
    }

    private void loadNormalizer() {
        try {
            InputStream is = context.getAssets().open(NORMALIZER_ASSET_NAME);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null)
                sb.append(line);
            reader.close();

            JSONObject json = new JSONObject(sb.toString());
            JSONArray imuMeanArr = json.getJSONArray("imu_mean");
            JSONArray imuScaleArr = json.getJSONArray("imu_scale");
            imuMean = new float[imuMeanArr.length()];
            imuScale = new float[imuScaleArr.length()];
            for (int i = 0; i < imuMeanArr.length(); i++) {
                imuMean[i] = (float) imuMeanArr.getDouble(i);
                imuScale[i] = (float) imuScaleArr.getDouble(i);
            }

            JSONArray magMeanArr = json.getJSONArray("mag_mean");
            JSONArray magScaleArr = json.getJSONArray("mag_scale");
            magMean = new float[magMeanArr.length()];
            magScale = new float[magScaleArr.length()];
            for (int i = 0; i < magMeanArr.length(); i++) {
                magMean[i] = (float) magMeanArr.getDouble(i);
                magScale[i] = (float) magScaleArr.getDouble(i);
            }

            normalizerLoaded = true;
            Log.i(TAG, "✓ Normalizer loaded");
        } catch (Exception e) {
            Log.w(TAG, "⚠ Normalizer not loaded: " + e.getMessage());
            normalizerLoaded = false;
        }
    }

    private byte[] loadModelBytesFromAssets() {
        try {
            InputStream is = context.getAssets().open(MODEL_ASSET_NAME);
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int len;
            while ((len = is.read(buf)) != -1)
                bos.write(buf, 0, len);
            is.close();
            return bos.toByteArray();
        } catch (IOException e) {
            Log.d(TAG, "Model asset not found: " + MODEL_ASSET_NAME);
            return null;
        }
    }

    /**
     * Copy an asset file to internal storage.
     */
    private void copyAssetToFile(String assetName, File destFile) throws IOException {
        InputStream is = context.getAssets().open(assetName);
        FileOutputStream fos = new FileOutputStream(destFile);
        byte[] buf = new byte[8192];
        int len;
        while ((len = is.read(buf)) != -1) {
            fos.write(buf, 0, len);
        }
        fos.close();
        is.close();
    }

    private byte[] readFileBytes(File file) {
        try {
            FileInputStream fis = new FileInputStream(file);
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int len;
            while ((len = fis.read(buf)) != -1)
                bos.write(buf, 0, len);
            fis.close();
            return bos.toByteArray();
        } catch (IOException e) {
            Log.e(TAG, "Failed to read model file: " + e.getMessage());
            return null;
        }
    }

    private void logModelDetails() {
        if (ortSession == null)
            return;
        try {
            Log.i(TAG, "Input: " + ortSession.getInputNames());
            Log.i(TAG, "Output: " + ortSession.getOutputNames());
        } catch (Exception e) {
            Log.w(TAG, "Could not log model details");
        }
    }

    public boolean isModelLoaded() {
        return isModelLoaded;
    }

    public static String[] getActivityLabels() {
        return ACTIVITY_LABELS.clone();
    }

    public void close() {
        try {
            if (ortSession != null) {
                ortSession.close();
                ortSession = null;
            }
            isModelLoaded = false;
            Log.d(TAG, "ONNX session closed");
        } catch (Exception e) {
            Log.e(TAG, "Error closing session: " + e.getMessage());
        }
    }

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
