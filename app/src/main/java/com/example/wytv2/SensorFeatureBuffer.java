package com.example.wytv2;

import java.util.LinkedList;
import java.util.Queue;

/**
 * Collects and buffers sensor data for ML model inference.
 * Maintains a rolling window of 50 timesteps with sensor features.
 */
public class SensorFeatureBuffer {

    private static final int WINDOW_SIZE = 50;
    private static final int FEATURE_DIM = 69;

    // Rolling buffer for sensor readings
    private Queue<SensorReading> buffer = new LinkedList<>();

    /**
     * Represents a single timestep of sensor data.
     */
    private static class SensorReading {
        float[] accelerometer; // x, y, z
        float[] gyroscope; // x, y, z
        float[] magnetometer; // x, y, z
        long timestamp;

        SensorReading(float[] accel, float[] gyro, float[] mag, long timestamp) {
            this.accelerometer = accel != null ? accel.clone() : new float[3];
            this.gyroscope = gyro != null ? gyro.clone() : new float[3];
            this.magnetometer = mag != null ? mag.clone() : new float[3];
            this.timestamp = timestamp;
        }
    }

    /**
     * Add a new sensor reading to the buffer.
     * Automatically removes oldest reading if buffer is full.
     */
    public void addReading(float[] accel, float[] gyro, float[] mag, long timestamp) {
        // Add new reading
        buffer.offer(new SensorReading(accel, gyro, mag, timestamp));

        // Remove oldest if buffer exceeds window size
        while (buffer.size() > WINDOW_SIZE) {
            buffer.poll();
        }
    }

    /**
     * Check if buffer has enough data for inference.
     */
    public boolean isReady() {
        return buffer.size() >= WINDOW_SIZE;
    }

    /**
     * Extract features for ML model inference.
     * Returns a 50x69 array of features.
     * 
     * Feature layout (simplified):
     * - Columns 0-2: Accelerometer (x, y, z)
     * - Columns 3-5: Gyroscope (x, y, z)
     * - Columns 6-8: Magnetometer (x, y, z)
     * - Columns 9-11: Magnitudes (accel, gyro, mag)
     * - Columns 12-68: Padded with zeros (for future features)
     */
    public float[][] extractFeatures() {
        if (!isReady()) {
            return null;
        }

        float[][] features = new float[WINDOW_SIZE][FEATURE_DIM];

        int row = 0;
        for (SensorReading reading : buffer) {
            // Raw sensor values (9 features)
            features[row][0] = reading.accelerometer[0];
            features[row][1] = reading.accelerometer[1];
            features[row][2] = reading.accelerometer[2];

            features[row][3] = reading.gyroscope[0];
            features[row][4] = reading.gyroscope[1];
            features[row][5] = reading.gyroscope[2];

            features[row][6] = reading.magnetometer[0];
            features[row][7] = reading.magnetometer[1];
            features[row][8] = reading.magnetometer[2];

            // Magnitude features (3 features)
            features[row][9] = magnitude(reading.accelerometer);
            features[row][10] = magnitude(reading.gyroscope);
            features[row][11] = magnitude(reading.magnetometer);

            // Remaining features (12-68) are already initialized to 0
            // These can be filled with derived features in the future

            row++;
        }

        return features;
    }

    /**
     * Calculate magnitude of a 3D vector.
     */
    private float magnitude(float[] vector) {
        return (float) Math.sqrt(
                vector[0] * vector[0] +
                        vector[1] * vector[1] +
                        vector[2] * vector[2]);
    }

    /**
     * Clear the buffer.
     */
    public void clear() {
        buffer.clear();
    }

    /**
     * Get current buffer size.
     */
    public int getSize() {
        return buffer.size();
    }
}
