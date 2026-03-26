package com.example.wytv2.wifi;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.wifi.ScanResult;
import android.net.wifi.WifiManager;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Service for WiFi RSSI scanning and variance analysis.
 * Used for localization and false positive detection.
 */
public class WiFiRSSIService {
    private static final String TAG = "WiFiRSSI";

    private Context context;
    private WifiManager wifiManager;
    private Handler scanHandler;
    private boolean isScanning = false;

    // Scan configuration
    private static final long SCAN_INTERVAL_MS = 2000; // 2 seconds
    private static final int BUFFER_SIZE = 10; // Keep last 10 scans

    // RSSI buffer: BSSID -> List of recent RSSI values
    private Map<String, List<Integer>> rssiBuffer = new HashMap<>();

    // Current scan results
    private List<WiFiReading> currentReadings = new ArrayList<>();

    // Listener interface
    public interface RSSIListener {
        void onRSSIUpdate(List<WiFiReading> readings, float variance);

        void onScanComplete(List<WiFiReading> readings);
    }

    private RSSIListener listener;

    // Broadcast receiver for scan results
    private BroadcastReceiver wifiScanReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            boolean success = intent.getBooleanExtra(WifiManager.EXTRA_RESULTS_UPDATED, false);
            if (success) {
                processScanResults();
            } else {
                Log.w(TAG, "WiFi scan failed");
            }
        }
    };

    public WiFiRSSIService(Context context) {
        this.context = context;
        this.wifiManager = (WifiManager) context.getApplicationContext()
                .getSystemService(Context.WIFI_SERVICE);
        this.scanHandler = new Handler(Looper.getMainLooper());
    }

    /**
     * Start periodic WiFi scanning.
     */
    public void startScanning() {
        if (isScanning) {
            return;
        }

        if (wifiManager == null) {
            Log.e(TAG, "WiFi manager not available");
            return;
        }

        isScanning = true;

        // Register broadcast receiver
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION);
        context.registerReceiver(wifiScanReceiver, intentFilter);

        // Start periodic scanning
        scanHandler.post(scanRunnable);

        Log.d(TAG, "WiFi scanning started");
    }

    /**
     * Stop WiFi scanning.
     */
    public void stopScanning() {
        if (!isScanning) {
            return;
        }

        isScanning = false;
        scanHandler.removeCallbacks(scanRunnable);

        try {
            context.unregisterReceiver(wifiScanReceiver);
        } catch (IllegalArgumentException e) {
            // Receiver not registered
        }

        Log.d(TAG, "WiFi scanning stopped");
    }

    /**
     * Periodic scan runnable.
     */
    private Runnable scanRunnable = new Runnable() {
        @Override
        public void run() {
            if (isScanning) {
                triggerScan();
                scanHandler.postDelayed(this, SCAN_INTERVAL_MS);
            }
        }
    };

    /**
     * Trigger a WiFi scan.
     */
    private void triggerScan() {
        if (wifiManager != null) {
            boolean success = wifiManager.startScan();
            if (!success) {
                Log.w(TAG, "Failed to start WiFi scan");
            }
        }
    }

    /**
     * Process scan results from WiFi manager.
     */
    private void processScanResults() {
        if (wifiManager == null) {
            return;
        }

        List<ScanResult> results = wifiManager.getScanResults();
        currentReadings.clear();

        long timestamp = System.currentTimeMillis();

        for (ScanResult result : results) {
            String bssid = result.BSSID;
            String ssid = result.SSID;
            int rssi = result.level;

            // Add to current readings
            currentReadings.add(new WiFiReading(bssid, ssid, rssi, timestamp));

            // Update RSSI buffer
            if (!rssiBuffer.containsKey(bssid)) {
                rssiBuffer.put(bssid, new ArrayList<>());
            }

            List<Integer> rssiHistory = rssiBuffer.get(bssid);
            rssiHistory.add(rssi);

            // Keep only last BUFFER_SIZE readings
            if (rssiHistory.size() > BUFFER_SIZE) {
                rssiHistory.remove(0);
            }
        }

        // Calculate overall variance
        float variance = calculateOverallVariance();

        // Notify listener
        if (listener != null) {
            listener.onScanComplete(currentReadings);
            listener.onRSSIUpdate(currentReadings, variance);
        }

        // ENHANCED LOGGING FOR FINGERPRINT COLLECTION
        Log.d(TAG, "═══════════════════════════════════════════════════════");
        Log.d(TAG, String.format("WiFi Scan Complete: %d access points detected", currentReadings.size()));
        Log.d(TAG, "───────────────────────────────────────────────────────");

        // Log each AP with details for fingerprint collection
        for (int i = 0; i < currentReadings.size(); i++) {
            WiFiReading reading = currentReadings.get(i);
            Log.d(TAG, String.format("AP %d: SSID=\"%s\" BSSID=%s RSSI=%d dBm",
                    i + 1, reading.ssid, reading.bssid, reading.rssi));
        }

        Log.d(TAG, "───────────────────────────────────────────────────────");
        Log.d(TAG, String.format("RSSI Variance: %.2f (stable: %s)",
                variance, variance < 5.0f ? "YES" : "NO"));
        Log.d(TAG, "═══════════════════════════════════════════════════════");

        // Original summary log (keep for compatibility)
        Log.d(TAG, String.format("Scan complete: %d APs, variance: %.2f",
                currentReadings.size(), variance));
    }

    /**
     * Calculate RSSI variance for a specific BSSID.
     */
    public float getRSSIVariance(String bssid) {
        if (!rssiBuffer.containsKey(bssid)) {
            return 0.0f;
        }

        List<Integer> rssiHistory = rssiBuffer.get(bssid);
        if (rssiHistory.size() < 2) {
            return 0.0f;
        }

        // Calculate mean
        float sum = 0;
        for (int rssi : rssiHistory) {
            sum += rssi;
        }
        float mean = sum / rssiHistory.size();

        // Calculate variance
        float varianceSum = 0;
        for (int rssi : rssiHistory) {
            float diff = rssi - mean;
            varianceSum += diff * diff;
        }

        return varianceSum / rssiHistory.size();
    }

    /**
     * Calculate overall RSSI variance across all APs.
     */
    private float calculateOverallVariance() {
        if (rssiBuffer.isEmpty()) {
            return 0.0f;
        }

        float totalVariance = 0;
        int count = 0;

        for (String bssid : rssiBuffer.keySet()) {
            float variance = getRSSIVariance(bssid);
            if (variance > 0) {
                totalVariance += variance;
                count++;
            }
        }

        return count > 0 ? totalVariance / count : 0.0f;
    }

    /**
     * Check if RSSI is stable (low variance = stationary).
     */
    public boolean isRSSIStable() {
        float variance = calculateOverallVariance();
        return variance < 5.0f; // Threshold for stability
    }

    /**
     * Get current WiFi readings.
     */
    public List<WiFiReading> getCurrentReadings() {
        return new ArrayList<>(currentReadings);
    }

    /**
     * Set RSSI update listener.
     */
    public void setRSSIListener(RSSIListener listener) {
        this.listener = listener;
    }

    /**
     * Clear RSSI buffer.
     */
    public void clearBuffer() {
        rssiBuffer.clear();
        currentReadings.clear();
    }
}
