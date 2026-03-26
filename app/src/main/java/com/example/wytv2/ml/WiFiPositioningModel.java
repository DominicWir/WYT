package com.example.wytv2.ml;

import android.content.Context;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * ML-enhanced WiFi positioning model using xLSTM trained on UJIIndoorLoc.
 *
 * The model takes a sequence of sensor feature windows and predicts (x, y, floor).
 * Feature extraction replicates the Python training pipeline:
 *   - IMU (6ch): 7 stats × 6 axes + 4 magnitude features = 46
 *   - Magnetic (3ch): 7 stats × 3 axes + 2 magnitude features = 23
 *   - WiFi: 7 aggregate stats + 1 AP count = 8
 *   - Total: 77 features per window
 *
 * Input shape:  [1, SEQUENCE_LENGTH, 77]
 * Output:       position [1, SEQUENCE_LENGTH, 3] → (longitude, latitude, floor)
 */
public class WiFiPositioningModel {
    private static final String TAG = "WiFiPositioning";

    private static final String MODEL_ASSET = "indoor_positioning.onnx";
    private static final String NORMALIZER_ASSET = "positioning_normalizer.json";

    // Must match training config
    private static final int WINDOW_SIZE = 50;     // Samples per feature window
    private static final int FEATURE_DIM = 77;     // Features per window
    private static final int SEQUENCE_LENGTH = 50;  // Windows per model input
    private static final int POSITION_DIM = 3;      // (x, y, floor)

    // Percentiles used in training
    private static final float[] PERCENTILES = {25.0f, 50.0f, 75.0f};

    // ONNX Runtime
    private OrtEnvironment ortEnv;
    private OrtSession ortSession;
    private boolean isModelLoaded = false;

    // Normalizer parameters
    private float[] imuMean, imuScale;
    private float[] magMean, magScale;
    private float[] wifiMean, wifiScale;
    private boolean normalizerLoaded = false;

    // Sensor data buffer (raw samples for one feature window)
    private final Queue<float[]> accelBuffer = new LinkedList<>();
    private final Queue<float[]> gyroBuffer = new LinkedList<>();
    private final Queue<float[]> magBuffer = new LinkedList<>();

    // Feature window buffer (sequence of extracted feature vectors)
    private final List<float[]> featureSequence = new ArrayList<>();

    // Latest WiFi RSSI values
    private float[] latestWifiRssi = null;
    private int latestWifiApCount = 0;

    // Latest prediction and displacement tracking
    private float[] lastPrediction = null;
    private float[] previousPrediction = null; // For computing deltas
    private long lastPredictionTime = 0;
    private int predictionCount = 0;
    private static final long PREDICTION_INTERVAL_MS = 5000; // Every 5 seconds

    private Context context;

    /**
     * Result of a positioning prediction, including relative displacement.
     */
    public static class PositionPrediction {
        public final float x;   // Raw ML coordinate (UJIIndoorLoc space)
        public final float y;
        public final int floor;
        public final long timestamp;

        // Relative displacement from previous prediction
        public final float deltaX;
        public final float deltaY;
        public final boolean hasDelta; // false for first prediction

        public PositionPrediction(float x, float y, int floor, long timestamp,
                                  float deltaX, float deltaY, boolean hasDelta) {
            this.x = x;
            this.y = y;
            this.floor = floor;
            this.timestamp = timestamp;
            this.deltaX = deltaX;
            this.deltaY = deltaY;
            this.hasDelta = hasDelta;
        }

        @Override
        public String toString() {
            if (hasDelta) {
                return String.format("ML Δ=(%.2f, %.2f) raw=(%.2f, %.2f) floor %d",
                        deltaX, deltaY, x, y, floor);
            }
            return String.format("ML Position: (%.2f, %.2f) floor %d [first]", x, y, floor);
        }
    }

    /**
     * Initialize the positioning model.
     */
    public boolean initialize(Context context) {
        this.context = context;
        loadNormalizer();

        try {
            ortEnv = OrtEnvironment.getEnvironment();

            // Copy model from assets to internal storage
            File modelDir = new File(context.getFilesDir(), "positioning_model");
            if (!modelDir.exists()) modelDir.mkdirs();

            File modelFile = new File(modelDir, MODEL_ASSET);
            if (!modelFile.exists() || modelFile.length() == 0) {
                copyAssetToFile(MODEL_ASSET, modelFile);
            }

            if (modelFile.exists() && modelFile.length() > 0) {
                ortSession = ortEnv.createSession(modelFile.getAbsolutePath());
                isModelLoaded = true;
                Log.i(TAG, String.format("✓ WiFi positioning model loaded (%d KB)",
                        modelFile.length() / 1024));
                logModelDetails();
            } else {
                Log.w(TAG, "⚠ No positioning model found");
                isModelLoaded = false;
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to load positioning model: " + e.getMessage(), e);
            isModelLoaded = false;
        }

        return isModelLoaded;
    }

    /**
     * Add a sensor reading to the buffer.
     * Call this from onSensorChanged at ~50Hz.
     */
    public void addSensorReading(float[] accel, float[] gyro, float[] mag) {
        accelBuffer.offer(accel.clone());
        gyroBuffer.offer(gyro.clone());
        magBuffer.offer(mag.clone());

        // Keep buffer at window size
        while (accelBuffer.size() > WINDOW_SIZE) {
            accelBuffer.poll();
            gyroBuffer.poll();
            magBuffer.poll();
        }

        // Extract features when we have a full window
        if (accelBuffer.size() >= WINDOW_SIZE) {
            float[] features = extractWindowFeatures();
            if (features != null) {
                featureSequence.add(features);
                // Keep sequence at max length
                while (featureSequence.size() > SEQUENCE_LENGTH) {
                    featureSequence.remove(0);
                }
            }
        }
    }

    /**
     * Update WiFi RSSI values (called less frequently, ~every 10s).
     */
    public void updateWifiReadings(float[] rssiValues, int apCount) {
        this.latestWifiRssi = rssiValues != null ? rssiValues.clone() : null;
        this.latestWifiApCount = apCount;
    }

    /**
     * Run positioning prediction if enough data and enough time has elapsed.
     * @return predicted position, or null if not ready
     */
    public PositionPrediction predict() {
        if (!isModelLoaded) return null;

        // Check timing
        long now = System.currentTimeMillis();
        if (now - lastPredictionTime < PREDICTION_INTERVAL_MS) return null;

        // Need full sequence
        if (featureSequence.size() < SEQUENCE_LENGTH) {
            Log.d(TAG, String.format("Buffering features: %d/%d windows",
                    featureSequence.size(), SEQUENCE_LENGTH));
            return null;
        }

        try {
            // Flatten feature sequence to [1, SEQUENCE_LENGTH, FEATURE_DIM]
            float[] flatInput = new float[SEQUENCE_LENGTH * FEATURE_DIM];
            for (int i = 0; i < SEQUENCE_LENGTH; i++) {
                float[] window = featureSequence.get(featureSequence.size() - SEQUENCE_LENGTH + i);
                System.arraycopy(window, 0, flatInput, i * FEATURE_DIM, FEATURE_DIM);
            }

            // Create ONNX tensor
            long[] shape = {1, SEQUENCE_LENGTH, FEATURE_DIM};
            OnnxTensor inputTensor = OnnxTensor.createTensor(
                    ortEnv, FloatBuffer.wrap(flatInput), shape);

            // Run inference
            OrtSession.Result result = ortSession.run(
                    Collections.singletonMap("input", inputTensor));

            // Get position output [1, SEQUENCE_LENGTH, 3]
            float[][][] positionOutput = (float[][][]) result.get(0).getValue();
            float[] latestPosition = positionOutput[0][SEQUENCE_LENGTH - 1]; // Last timestep

            // Compute relative displacement from previous prediction
            float deltaX = 0, deltaY = 0;
            boolean hasDelta = (previousPrediction != null);
            if (hasDelta) {
                deltaX = latestPosition[0] - previousPrediction[0];
                deltaY = latestPosition[1] - previousPrediction[1];
            }

            previousPrediction = lastPrediction;
            lastPrediction = latestPosition.clone();
            lastPredictionTime = now;
            predictionCount++;

            // Clean up
            inputTensor.close();
            result.close();

            PositionPrediction prediction = new PositionPrediction(
                    latestPosition[0], latestPosition[1],
                    Math.round(latestPosition[2]), now,
                    deltaX, deltaY, hasDelta);

            if (hasDelta) {
                Log.d(TAG, String.format(
                        "Prediction #%d: raw=(%.2f, %.2f) Δ=(%.4f, %.4f) floor %.1f",
                        predictionCount, latestPosition[0], latestPosition[1],
                        deltaX, deltaY, latestPosition[2]));
            } else {
                Log.d(TAG, String.format(
                        "Prediction #%d [baseline]: raw=(%.2f, %.2f) floor %.1f",
                        predictionCount, latestPosition[0], latestPosition[1], latestPosition[2]));
            }

            return prediction;

        } catch (Exception e) {
            Log.e(TAG, "Inference failed: " + e.getMessage());
            return null;
        }
    }

    // ========================================
    // Feature Extraction (matches Python pipeline)
    // ========================================

    /**
     * Extract 77 statistical features from the current sensor window.
     * Replicates SensorFeatureExtractor.extract_window_features() from training.
     */
    private float[] extractWindowFeatures() {
        float[][] accelArr = bufferToArray(accelBuffer, 3);
        float[][] gyroArr = bufferToArray(gyroBuffer, 3);
        float[][] magArr = bufferToArray(magBuffer, 3);

        if (accelArr == null || gyroArr == null || magArr == null) return null;

        List<Float> features = new ArrayList<>();

        // --- IMU features (46 total) ---
        // Combine accel + gyro into 6-channel IMU
        float[][] imu = new float[WINDOW_SIZE][6];
        for (int i = 0; i < WINDOW_SIZE; i++) {
            // Normalize if normalizer is loaded
            for (int j = 0; j < 3; j++) {
                imu[i][j] = normalizerLoaded
                        ? (accelArr[i][j] - imuMean[j]) / imuScale[j]
                        : accelArr[i][j];
                imu[i][j + 3] = normalizerLoaded
                        ? (gyroArr[i][j] - imuMean[j + 3]) / imuScale[j + 3]
                        : gyroArr[i][j];
            }
        }

        // Statistical features for each of 6 IMU channels: min, max, mean, std, p25, p50, p75
        for (int ch = 0; ch < 6; ch++) {
            float[] col = new float[WINDOW_SIZE];
            for (int i = 0; i < WINDOW_SIZE; i++) col[i] = imu[i][ch];
            addStatisticalFeatures(features, col);
        }

        // Accelerometer magnitude features (mean, std)
        float[] accMag = new float[WINDOW_SIZE];
        for (int i = 0; i < WINDOW_SIZE; i++) {
            accMag[i] = (float) Math.sqrt(
                    imu[i][0] * imu[i][0] + imu[i][1] * imu[i][1] + imu[i][2] * imu[i][2]);
        }
        features.add(mean(accMag));
        features.add(std(accMag));

        // Gyroscope magnitude features (mean, std)
        float[] gyroMag = new float[WINDOW_SIZE];
        for (int i = 0; i < WINDOW_SIZE; i++) {
            gyroMag[i] = (float) Math.sqrt(
                    imu[i][3] * imu[i][3] + imu[i][4] * imu[i][4] + imu[i][5] * imu[i][5]);
        }
        features.add(mean(gyroMag));
        features.add(std(gyroMag));

        // --- Magnetic features (23 total) ---
        float[][] magNorm = new float[WINDOW_SIZE][3];
        for (int i = 0; i < WINDOW_SIZE; i++) {
            for (int j = 0; j < 3; j++) {
                magNorm[i][j] = normalizerLoaded
                        ? (magArr[i][j] - magMean[j]) / magScale[j]
                        : magArr[i][j];
            }
        }

        for (int ch = 0; ch < 3; ch++) {
            float[] col = new float[WINDOW_SIZE];
            for (int i = 0; i < WINDOW_SIZE; i++) col[i] = magNorm[i][ch];
            addStatisticalFeatures(features, col);
        }

        // Magnetic magnitude features
        float[] magMagnitude = new float[WINDOW_SIZE];
        for (int i = 0; i < WINDOW_SIZE; i++) {
            magMagnitude[i] = (float) Math.sqrt(
                    magNorm[i][0] * magNorm[i][0] +
                    magNorm[i][1] * magNorm[i][1] +
                    magNorm[i][2] * magNorm[i][2]);
        }
        features.add(mean(magMagnitude));
        features.add(std(magMagnitude));

        // --- WiFi features (8 total) ---
        if (latestWifiRssi != null && latestWifiRssi.length > 0) {
            // Normalize WiFi if normalizer loaded
            float[] wifiNorm;
            if (normalizerLoaded && wifiMean != null) {
                wifiNorm = new float[latestWifiRssi.length];
                for (int i = 0; i < latestWifiRssi.length; i++) {
                    int idx = i < wifiMean.length ? i : wifiMean.length - 1;
                    wifiNorm[i] = (latestWifiRssi[i] - wifiMean[idx]) / wifiScale[idx];
                }
            } else {
                wifiNorm = latestWifiRssi;
            }
            addStatisticalFeatures(features, wifiNorm);
            features.add((float) latestWifiApCount);
        } else {
            // No WiFi data — zeros
            for (int i = 0; i < 8; i++) features.add(0.0f);
        }

        // Convert to float array
        float[] result = new float[features.size()];
        for (int i = 0; i < features.size(); i++) {
            result[i] = features.get(i);
        }

        // Pad or truncate to FEATURE_DIM
        if (result.length != FEATURE_DIM) {
            float[] padded = new float[FEATURE_DIM];
            System.arraycopy(result, 0, padded, 0, Math.min(result.length, FEATURE_DIM));
            return padded;
        }

        return result;
    }

    /**
     * Add min, max, mean, std, p25, p50, p75 for a data column.
     */
    private void addStatisticalFeatures(List<Float> features, float[] data) {
        float[] sorted = data.clone();
        Arrays.sort(sorted);

        features.add(sorted[0]);                    // min
        features.add(sorted[sorted.length - 1]);    // max
        features.add(mean(data));                    // mean
        features.add(std(data));                     // std

        // Percentiles
        for (float p : PERCENTILES) {
            float idx = (p / 100.0f) * (sorted.length - 1);
            int lower = (int) Math.floor(idx);
            int upper = Math.min(lower + 1, sorted.length - 1);
            float frac = idx - lower;
            features.add(sorted[lower] * (1 - frac) + sorted[upper] * frac);
        }
    }

    // ========================================
    // Utility methods
    // ========================================

    private float mean(float[] data) {
        float sum = 0;
        for (float v : data) sum += v;
        return sum / data.length;
    }

    private float std(float[] data) {
        float m = mean(data);
        float sumSq = 0;
        for (float v : data) sumSq += (v - m) * (v - m);
        return (float) Math.sqrt(sumSq / data.length);
    }

    private float[][] bufferToArray(Queue<float[]> buffer, int dim) {
        if (buffer.size() < WINDOW_SIZE) return null;
        float[][] arr = new float[WINDOW_SIZE][dim];
        int i = 0;
        for (float[] vals : buffer) {
            if (i >= WINDOW_SIZE) break;
            System.arraycopy(vals, 0, arr[i], 0, Math.min(vals.length, dim));
            i++;
        }
        return arr;
    }

    private void loadNormalizer() {
        try {
            InputStream is = context.getAssets().open(NORMALIZER_ASSET);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) sb.append(line);
            reader.close();

            JSONObject json = new JSONObject(sb.toString());

            imuMean = jsonArrayToFloat(json.getJSONArray("imu_mean"));
            imuScale = jsonArrayToFloat(json.getJSONArray("imu_scale"));
            magMean = jsonArrayToFloat(json.getJSONArray("mag_mean"));
            magScale = jsonArrayToFloat(json.getJSONArray("mag_scale"));

            if (json.has("wifi_mean")) {
                wifiMean = jsonArrayToFloat(json.getJSONArray("wifi_mean"));
                wifiScale = jsonArrayToFloat(json.getJSONArray("wifi_scale"));
            }

            normalizerLoaded = true;
            Log.i(TAG, "✓ Positioning normalizer loaded");
        } catch (Exception e) {
            Log.w(TAG, "⚠ Normalizer not loaded: " + e.getMessage());
            normalizerLoaded = false;
        }
    }

    private float[] jsonArrayToFloat(JSONArray arr) throws Exception {
        float[] result = new float[arr.length()];
        for (int i = 0; i < arr.length(); i++) {
            result[i] = (float) arr.getDouble(i);
        }
        return result;
    }

    private void copyAssetToFile(String assetName, File destFile) throws IOException {
        InputStream is = context.getAssets().open(assetName);
        FileOutputStream fos = new FileOutputStream(destFile);
        byte[] buf = new byte[8192];
        int len;
        while ((len = is.read(buf)) != -1) fos.write(buf, 0, len);
        fos.close();
        is.close();
    }

    private void logModelDetails() {
        if (ortSession == null) return;
        try {
            Log.i(TAG, "Input names: " + ortSession.getInputNames());
            Log.i(TAG, "Output names: " + ortSession.getOutputNames());
        } catch (Exception e) {
            Log.w(TAG, "Could not log model details");
        }
    }

    public boolean isModelLoaded() {
        return isModelLoaded;
    }

    public float[] getLastPrediction() {
        return lastPrediction;
    }

    public void close() {
        try {
            if (ortSession != null) {
                ortSession.close();
                ortSession = null;
            }
            isModelLoaded = false;
            Log.d(TAG, "Positioning model session closed");
        } catch (Exception e) {
            Log.e(TAG, "Error closing session: " + e.getMessage());
        }
    }
}
