package com.example.wytv2.pdr;

import android.util.Log;
import com.example.wytv2.wifi.WiFiFingerprintDatabase;
import com.example.wytv2.wifi.WiFiReading;
import com.example.wytv2.pdr.magneticfield.MagneticFieldLocalizationService;

import java.util.ArrayList;
import java.util.List;

/**
 * Helper class to load sample fingerprints for testing 3-sensor fusion.
 * This populates WiFi and magnetic databases with hardcoded test data.
 */
public class SampleFingerprintLoader {
    private static final String TAG = "SampleFingerprints";

    /**
     * Load sample WiFi and magnetic fingerprints into the databases.
     * 
     * @param wifiDatabase    WiFi fingerprint database to populate
     * @param magneticService Magnetic localization service to populate
     */
    public static void loadSamples(WiFiFingerprintDatabase wifiDatabase,
            MagneticFieldLocalizationService magneticService) {
        Log.d(TAG, "Loading sample fingerprints for testing...");

        long timestamp = System.currentTimeMillis();

        // TEMPORARILY DISABLED: WiFi fingerprints don't match actual walking
        // environment
        // Current fingerprints are along X-axis (0,0), (5,0), (10,0)
        // User walks along Y-axis, causing position jumps
        // Previous issue: WiFi at (5,0) pulled position from (-0.0,-1.6) to
        // (3.31,-0.97)
        // TODO: Collect real WiFi fingerprints at actual walking positions
        /*
         * // Sample WiFi fingerprints at 3 positions
         * // Position 1: Origin (0, 0)
         * List<WiFiReading> wifi1 = new ArrayList<>();
         * wifi1.add(new WiFiReading("00:11:22:33:44:01", "TestAP1", -45, timestamp));
         * wifi1.add(new WiFiReading("00:11:22:33:44:02", "TestAP2", -65, timestamp));
         * wifiDatabase.addFingerprint(0.0f, 0.0f, 1, wifi1);
         * 
         * // Position 2: (5, 0) - 5m east
         * List<WiFiReading> wifi2 = new ArrayList<>();
         * wifi2.add(new WiFiReading("00:11:22:33:44:01", "TestAP1", -55, timestamp));
         * wifi2.add(new WiFiReading("00:11:22:33:44:02", "TestAP2", -50, timestamp));
         * wifiDatabase.addFingerprint(5.0f, 0.0f, 1, wifi2);
         * 
         * // Position 3: (10, 0) - 10m east
         * List<WiFiReading> wifi3 = new ArrayList<>();
         * wifi3.add(new WiFiReading("00:11:22:33:44:01", "TestAP1", -70, timestamp));
         * wifi3.add(new WiFiReading("00:11:22:33:44:02", "TestAP2", -45, timestamp));
         * wifiDatabase.addFingerprint(10.0f, 0.0f, 1, wifi3);
         */

        // Sample Magnetic fingerprints at same 3 positions
        // UPDATED: Using actual magnetic readings from device to ensure correct
        // position matching
        // These values are based on observed device readings during testing
        // The magnetic service will match current readings to these fingerprints
        // and return the associated positions: (0,0), (5,0), (10,0)

        float[] mag1 = { 43.3f, 31.8f, 20.0f }; // Origin (0,0) - matches observed readings
        magneticService.addFingerprint(0.0f, 0.0f, 1, mag1);

        float[] mag2 = { 45.9f, 32.5f, 22.0f }; // 5m east (5,0) - intermediate values
        magneticService.addFingerprint(5.0f, 0.0f, 1, mag2);

        float[] mag3 = { 69.4f, 16.9f, 24.0f }; // 10m east (10,0) - matches observed readings
        magneticService.addFingerprint(10.0f, 0.0f, 1, mag3);

        Log.d(TAG, "✓ Loaded 3 magnetic sample fingerprints (WiFi disabled)");
        Log.d(TAG, "  Positions: (0,0), (5,0), (10,0) on floor 1");
        Log.d(TAG, "  Magnetic values updated to match device readings");
        Log.d(TAG, "  WiFi fingerprints disabled - awaiting real data collection");
    }
}
