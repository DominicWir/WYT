package com.example.wytv2.wifi;

import android.util.Log;

import java.util.ArrayList;
import java.util.List;

/**
 * WiFi fingerprint database for indoor localization using k-NN matching.
 */
public class WiFiFingerprintDatabase {
    private static final String TAG = "WiFiFingerprint";

    private List<Fingerprint> fingerprints = new ArrayList<>();

    /**
     * Represents a WiFi fingerprint at a specific location.
     */
    public static class Fingerprint {
        public float x;
        public float y;
        public int floor;
        public List<WiFiReading> readings;
        public long timestamp;

        public Fingerprint(float x, float y, int floor, List<WiFiReading> readings) {
            this.x = x;
            this.y = y;
            this.floor = floor;
            this.readings = new ArrayList<>(readings);
            this.timestamp = System.currentTimeMillis();
        }
    }

    /**
     * Estimated position from k-NN matching.
     */
    public static class Position {
        public float x;
        public float y;
        public int floor;
        public float accuracy; // Estimated error in meters

        public Position(float x, float y, int floor, float accuracy) {
            this.x = x;
            this.y = y;
            this.floor = floor;
            this.accuracy = accuracy;
        }
    }

    /**
     * Add a fingerprint to the database.
     */
    public void addFingerprint(float x, float y, int floor, List<WiFiReading> readings) {
        if (readings == null || readings.isEmpty()) {
            Log.w(TAG, "Cannot add fingerprint with no readings");
            return;
        }

        Fingerprint fp = new Fingerprint(x, y, floor, readings);
        fingerprints.add(fp);

        Log.d(TAG, String.format("Added fingerprint at (%.1f, %.1f) floor %d with %d APs",
                x, y, floor, readings.size()));
    }

    /**
     * Estimate position using k-NN matching.
     */
    public Position estimatePosition(List<WiFiReading> currentScan, int k) {
        if (fingerprints.isEmpty()) {
            Log.w(TAG, "No fingerprints in database");
            return null;
        }

        if (currentScan == null || currentScan.isEmpty()) {
            Log.w(TAG, "No current scan data");
            return null;
        }

        // Calculate distances to all fingerprints
        List<FingerprintDistance> distances = new ArrayList<>();

        for (Fingerprint fp : fingerprints) {
            float distance = calculateDistance(currentScan, fp.readings);
            distances.add(new FingerprintDistance(fp, distance));
        }

        // Sort by distance
        distances.sort((a, b) -> Float.compare(a.distance, b.distance));

        // Take k nearest neighbors
        int numNeighbors = Math.min(k, distances.size());

        // Weighted average of positions
        float totalWeight = 0;
        float weightedX = 0;
        float weightedY = 0;
        int mostCommonFloor = 0;

        for (int i = 0; i < numNeighbors; i++) {
            FingerprintDistance fd = distances.get(i);

            // Weight inversely proportional to distance
            float weight = 1.0f / (1.0f + fd.distance);
            totalWeight += weight;

            weightedX += fd.fingerprint.x * weight;
            weightedY += fd.fingerprint.y * weight;

            // Use floor from nearest neighbor
            if (i == 0) {
                mostCommonFloor = fd.fingerprint.floor;
            }
        }

        float estimatedX = weightedX / totalWeight;
        float estimatedY = weightedY / totalWeight;

        // Estimate accuracy based on nearest neighbor distance
        // Clamp to reasonable range to prevent overflow
        float nearestDistance = distances.get(0).distance;
        float accuracy = nearestDistance == Float.MAX_VALUE ? 10.0f
                : Math.min(10.0f, Math.max(0.5f, nearestDistance / 10.0f));

        Position pos = new Position(estimatedX, estimatedY, mostCommonFloor, accuracy);

        Log.d(TAG, String.format("Estimated position: (%.1f, %.1f) floor %d, accuracy: ±%.1fm",
                pos.x, pos.y, pos.floor, pos.accuracy));

        return pos;
    }

    /**
     * Calculate Euclidean distance in RSSI space.
     */
    private float calculateDistance(List<WiFiReading> scan1, List<WiFiReading> scan2) {
        // Create BSSID -> RSSI maps
        java.util.Map<String, Integer> map1 = new java.util.HashMap<>();
        java.util.Map<String, Integer> map2 = new java.util.HashMap<>();

        for (WiFiReading r : scan1) {
            map1.put(r.bssid, r.rssi);
        }

        for (WiFiReading r : scan2) {
            map2.put(r.bssid, r.rssi);
        }

        // Find common BSSIDs
        java.util.Set<String> commonBSSIDs = new java.util.HashSet<>(map1.keySet());
        commonBSSIDs.retainAll(map2.keySet());

        if (commonBSSIDs.isEmpty()) {
            return Float.MAX_VALUE; // No common APs
        }

        // Calculate Euclidean distance
        float sumSquaredDiff = 0;

        for (String bssid : commonBSSIDs) {
            int rssi1 = map1.get(bssid);
            int rssi2 = map2.get(bssid);
            float diff = rssi1 - rssi2;
            sumSquaredDiff += diff * diff;
        }

        return (float) Math.sqrt(sumSquaredDiff / commonBSSIDs.size());
    }

    /**
     * Helper class for sorting fingerprints by distance.
     */
    private static class FingerprintDistance {
        Fingerprint fingerprint;
        float distance;

        FingerprintDistance(Fingerprint fingerprint, float distance) {
            this.fingerprint = fingerprint;
            this.distance = distance;
        }
    }

    /**
     * Get number of fingerprints in database.
     */
    public int size() {
        return fingerprints.size();
    }

    /**
     * Clear all fingerprints.
     */
    public void clear() {
        fingerprints.clear();
        Log.d(TAG, "Fingerprint database cleared");
    }
}
