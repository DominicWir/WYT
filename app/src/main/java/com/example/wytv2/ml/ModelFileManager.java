package com.example.wytv2.ml;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Manages model file storage, versioning, and persistence.
 * Handles saving retrained models and loading them on app startup.
 */
public class ModelFileManager {
    private static final String TAG = "ModelFileManager";

    private static final String MODEL_DIR = "models";
    private static final String BASE_MODEL = "activity_model.onnx";
    private static final String CURRENT_MODEL = "activity_model_current.onnx";
    private static final String BACKUP_MODEL = "activity_model_backup.onnx";

    private static final String PREFS_NAME = "ModelVersion";
    private static final String KEY_VERSION = "version";
    private static final String KEY_LAST_UPDATE = "last_update";
    private static final String KEY_MODEL_TYPE = "model_type";

    private Context context;
    private File modelDirectory;

    public ModelFileManager(Context context) {
        this.context = context;
        this.modelDirectory = new File(context.getFilesDir(), MODEL_DIR);

        // Create model directory if it doesn't exist
        if (!modelDirectory.exists()) {
            boolean created = modelDirectory.mkdirs();
            Log.d(TAG, "Model directory created: " + created);
        }
    }

    /**
     * Get path to current active model.
     * Returns base model if no retrained model exists.
     */
    public File getCurrentModelPath() {
        File currentModel = new File(modelDirectory, CURRENT_MODEL);

        if (currentModel.exists()) {
            Log.d(TAG, "Using retrained model: " + currentModel.getAbsolutePath());
            return currentModel;
        }

        // Fall back to base model from assets
        Log.d(TAG, "No retrained model found, using base model");
        return copyBaseModelFromAssets();
    }

    /**
     * Save a newly retrained model.
     * Creates backup of previous model before overwriting.
     * 
     * @param newModelFile The retrained model file to save
     * @return true if successful, false otherwise
     */
    public boolean saveRetrainedModel(File newModelFile) {
        if (newModelFile == null || !newModelFile.exists()) {
            Log.e(TAG, "Invalid model file provided");
            return false;
        }

        try {
            File currentModel = new File(modelDirectory, CURRENT_MODEL);

            // Backup existing model if it exists
            if (currentModel.exists()) {
                File backup = new File(modelDirectory, BACKUP_MODEL);
                copyFile(currentModel, backup);
                Log.d(TAG, "Previous model backed up");
            }

            // Save new model
            copyFile(newModelFile, currentModel);
            Log.d(TAG, "Retrained model saved successfully: " + currentModel.getAbsolutePath());

            return true;
        } catch (IOException e) {
            Log.e(TAG, "Failed to save retrained model", e);
            return false;
        }
    }

    /**
     * Restore previous model (rollback).
     * Useful if new model performs worse than previous.
     * 
     * @return true if rollback successful, false if no backup exists
     */
    public boolean restorePreviousModel() {
        File backup = new File(modelDirectory, BACKUP_MODEL);
        File current = new File(modelDirectory, CURRENT_MODEL);

        if (!backup.exists()) {
            Log.w(TAG, "No backup model available for rollback");
            return false;
        }

        try {
            copyFile(backup, current);
            Log.d(TAG, "Rolled back to previous model");

            // Decrement version
            SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
            int currentVersion = prefs.getInt(KEY_VERSION, 0);
            prefs.edit()
                    .putInt(KEY_VERSION, Math.max(0, currentVersion - 1))
                    .putLong(KEY_LAST_UPDATE, System.currentTimeMillis())
                    .apply();

            return true;
        } catch (IOException e) {
            Log.e(TAG, "Failed to rollback model", e);
            return false;
        }
    }

    /**
     * Get model version information.
     * 
     * @return ModelVersionInfo containing type, version, and last update time
     */
    public ModelVersionInfo getVersionInfo() {
        File currentModel = new File(modelDirectory, CURRENT_MODEL);

        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);

        if (!currentModel.exists()) {
            // Using base model
            return new ModelVersionInfo("base", 0, 0);
        }

        // Using retrained model
        int version = prefs.getInt(KEY_VERSION, 0);
        long lastUpdate = prefs.getLong(KEY_LAST_UPDATE, 0);

        return new ModelVersionInfo("retrained", version, lastUpdate);
    }

    /**
     * Update version info after successful retraining.
     * Increments version number and updates timestamp.
     */
    public void updateVersionInfo() {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        int currentVersion = prefs.getInt(KEY_VERSION, 0);

        prefs.edit()
                .putInt(KEY_VERSION, currentVersion + 1)
                .putLong(KEY_LAST_UPDATE, System.currentTimeMillis())
                .putString(KEY_MODEL_TYPE, "retrained")
                .apply();

        Log.d(TAG, "Model version updated to: " + (currentVersion + 1));
    }

    /**
     * Reset to base model (clear all retrained models).
     * Useful for testing or if user wants to start fresh.
     */
    public void resetToBaseModel() {
        File currentModel = new File(modelDirectory, CURRENT_MODEL);
        File backup = new File(modelDirectory, BACKUP_MODEL);

        // Delete retrained models
        if (currentModel.exists()) {
            currentModel.delete();
        }
        if (backup.exists()) {
            backup.delete();
        }

        // Reset version info
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        prefs.edit()
                .putInt(KEY_VERSION, 0)
                .putLong(KEY_LAST_UPDATE, 0)
                .putString(KEY_MODEL_TYPE, "base")
                .apply();

        Log.d(TAG, "Reset to base model");
    }

    /**
     * Copy base model from assets to internal storage.
     * This is called when no retrained model exists.
     * Also copies companion .data file if present (ONNX external data).
     *
     * @return File pointing to the base model in internal storage
     */
    private File copyBaseModelFromAssets() {
        File baseModel = new File(modelDirectory, BASE_MODEL);

        // If already copied (both model and companion data), return it
        File dataFile = new File(modelDirectory, BASE_MODEL + ".data");
        boolean hasDataInAssets = false;
        try {
            context.getAssets().open(BASE_MODEL + ".data").close();
            hasDataInAssets = true;
        } catch (IOException e) {
            // No .data file in assets, model is self-contained
        }

        if (baseModel.exists() && (!hasDataInAssets || dataFile.exists())) {
            return baseModel;
        }

        // Copy main model from assets
        try {
            copyAsset(BASE_MODEL, baseModel);
            Log.d(TAG, "Base model copied from assets");
        } catch (IOException e) {
            Log.d(TAG, "No base model in assets (using rule-based model)");
        }

        // Copy companion .data file if it exists
        try {
            copyAsset(BASE_MODEL + ".data", dataFile);
            Log.d(TAG, "Companion .data file copied from assets");
        } catch (IOException e) {
            Log.d(TAG, "No companion .data file (model is self-contained)");
        }

        return baseModel;
    }

    /**
     * Copy a single asset file to a destination.
     */
    private void copyAsset(String assetName, File dest) throws IOException {
        InputStream is = context.getAssets().open(assetName);
        OutputStream os = new FileOutputStream(dest);
        byte[] buffer = new byte[4096];
        int length;
        while ((length = is.read(buffer)) > 0) {
            os.write(buffer, 0, length);
        }
        os.flush();
        os.close();
        is.close();
    }

    /**
     * Copy file from source to destination.
     * 
     * @param src Source file
     * @param dst Destination file
     * @throws IOException if copy fails
     */
    private void copyFile(File src, File dst) throws IOException {
        if (!src.exists()) {
            throw new IOException("Source file does not exist: " + src.getAbsolutePath());
        }

        FileInputStream fis = new FileInputStream(src);
        FileOutputStream fos = new FileOutputStream(dst);

        byte[] buffer = new byte[4096];
        int length;
        while ((length = fis.read(buffer)) > 0) {
            fos.write(buffer, 0, length);
        }

        fos.flush();
        fos.close();
        fis.close();
    }

    /**
     * Model version information.
     */
    public static class ModelVersionInfo {
        public final String type; // "base" or "retrained"
        public final int version; // Version number (0 for base)
        public final long lastUpdate; // Timestamp of last update

        public ModelVersionInfo(String type, int version, long lastUpdate) {
            this.type = type;
            this.version = version;
            this.lastUpdate = lastUpdate;
        }

        @Override
        public String toString() {
            if (version == 0) {
                return "Base model (v0)";
            }

            long ageMs = System.currentTimeMillis() - lastUpdate;
            long ageHours = ageMs / (1000 * 60 * 60);

            return String.format("Retrained model (v%d, updated %dh ago)",
                    version, ageHours);
        }
    }
}
