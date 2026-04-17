package com.example.wytv2.pdr;

import android.util.Log;
import com.example.wytv2.wifi.WiFiFingerprintDatabase;
import com.example.wytv2.pdr.magneticfield.MagneticFieldLocalizationService;

/**
 * Loads reference magnetic fingerprints used by the P0–P3 stationary correction
 * chain. WiFi fingerprints are populated from the UJIIndoorLoc-trained ONNX model
 * at runtime and do not require manual seeding.
 */
public class SampleFingerprintLoader {
    private static final String TAG = "SampleFingerprints";

    /**
     * Seeds the magnetic localization service with reference fingerprints.
     * These values correspond to observed magnetic readings during initial calibration.
     *
     * @param wifiDatabase    WiFi fingerprint database (populated by the ML model at runtime)
     * @param magneticService Magnetic localization service to seed
     */
    public static void loadSamples(WiFiFingerprintDatabase wifiDatabase,
            MagneticFieldLocalizationService magneticService) {

        // Magnetic reference fingerprints at three corridor positions.
        // Values are taken from device readings during the initial calibration walk.
        float[] mag1 = { 43.3f, 31.8f, 20.0f }; // Origin (0, 0)
        magneticService.addFingerprint(0.0f, 0.0f, 1, mag1);

        float[] mag2 = { 45.9f, 32.5f, 22.0f }; // 5 m east (5, 0)
        magneticService.addFingerprint(5.0f, 0.0f, 1, mag2);

        float[] mag3 = { 69.4f, 16.9f, 24.0f }; // 10 m east (10, 0)
        magneticService.addFingerprint(10.0f, 0.0f, 1, mag3);

        Log.d(TAG, "Loaded 3 magnetic reference fingerprints at (0,0), (5,0), (10,0) on floor 1.");
    }
}
