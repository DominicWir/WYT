package com.example.wytv2.pdr.magneticfield;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;
import java.util.ArrayList;
import java.util.List;

public class MagneticFieldLocalizationService {

    private SensorManager sensorManager;
    private Sensor magnetometer;

    // For collecting magnetic field data during first 10 steps
    private List<float[]> magneticSequence = new ArrayList<>();
    private boolean isCollecting = false;
    private int stepCounter = 0;

    // Parameters from the paper
    private static final int REQUIRED_STEPS = 10;

    // Gyroscope data for rotation normalization
    private float[] gyroData = new float[3];
    private long lastGyroTime = 0;
    private float[] orientation = new float[3]; // Roll, pitch, yaw in radians

    // KNN parameters
    private int kValue = 3; // As per paper
    private List<MagneticFingerprint> fingerprintDatabase;

    // Current building floor map
    private FloorMap currentFloorMap;

    // Interface for position updates
    public interface PositionEstimationListener {
        void onInitialPositionEstimated(float x, float y, int floor);

        void onCalibrationNodeMatched(float x, float y);

        void onMagneticDataCollected(float[] magneticVector);
    }

    private PositionEstimationListener positionListener;

    // Data structures
    public static class MagneticFingerprint {
        public float x; // X coordinate in meters
        public float y; // Y coordinate in meters
        public int floor; // Floor number
        public float[] magneticVector; // 3-axis magnetic field [Bx, By, Bz]

        public MagneticFingerprint(float x, float y, int floor, float[] magneticVector) {
            this.x = x;
            this.y = y;
            this.floor = floor;
            this.magneticVector = magneticVector.clone();
        }
    }

    public static class FloorMap {
        public int floorNumber;
        public String buildingName;
        public List<MagneticFingerprint> fingerprints = new ArrayList<>();
        public List<CalibrationNode> calibrationNodes = new ArrayList<>();

        public FloorMap(int floorNumber, String buildingName) {
            this.floorNumber = floorNumber;
            this.buildingName = buildingName;
        }
    }

    public static class CalibrationNode {
        public float x, y;
        public int floor;
        public float[] magneticVector;
        public String type; // "staircase", "elevator", "landmark"
        public String transitionId; // e.g., "S1", "E1" as in paper

        public CalibrationNode(float x, float y, int floor, float[] magneticVector,
                String type, String transitionId) {
            this.x = x;
            this.y = y;
            this.floor = floor;
            this.magneticVector = magneticVector.clone();
            this.type = type;
            this.transitionId = transitionId;
        }
    }

    public MagneticFieldLocalizationService(Context context) {
        sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        }

        // Initialize with empty database (to be loaded from file)
        fingerprintDatabase = new ArrayList<>();

        // Initialize orientation
        orientation[0] = 0; // roll
        orientation[1] = 0; // pitch
        orientation[2] = 0; // yaw
    }

    /**
     * Add a magnetic fingerprint to the database.
     * Used for testing and manual fingerprint collection.
     */
    public void addFingerprint(float x, float y, int floor, float[] magneticVector) {
        if (magneticVector == null || magneticVector.length < 3) {
            Log.w("MagneticLocalization", "Invalid magnetic vector");
            return;
        }

        MagneticFingerprint fp = new MagneticFingerprint(x, y, floor, magneticVector);
        fingerprintDatabase.add(fp);

        Log.d("MagneticLocalization", String.format(
                "Added magnetic fingerprint at (%.1f, %.1f) floor %d, mag: [%.1f, %.1f, %.1f]",
                x, y, floor, magneticVector[0], magneticVector[1], magneticVector[2]));
    }

    // Called from StepDetectionService when magnetic data is available
    public void onMagneticData(float[] magneticValues, long timestamp) {
        if (isCollecting) {
            // Apply rotation normalization (as per paper equation 13)
            float[] normalizedMagnetic = normalizeMagneticVector(magneticValues);

            magneticSequence.add(normalizedMagnetic);

            // Notify UI for visualization
            if (positionListener != null) {
                positionListener.onMagneticDataCollected(normalizedMagnetic);
            }

            // Log magnetic field strength for debugging
            float magnitude = calculateMagneticMagnitude(normalizedMagnetic);
            Log.d("MagneticLocalization",
                    String.format("Magnetic: [%.1f, %.1f, %.1f] μT, Magnitude: %.1f μT",
                            normalizedMagnetic[0], normalizedMagnetic[1],
                            normalizedMagnetic[2], magnitude));
        }
    }

    // Update gyroscope data for rotation calculation
    public void updateGyroscopeData(float[] gyroValues, long timestamp) {
        if (lastGyroTime == 0) {
            lastGyroTime = timestamp;
            return;
        }

        float dt = (timestamp - lastGyroTime) / 1000.0f; // Convert to seconds
        lastGyroTime = timestamp;

        // Simple integration of angular velocity to get orientation
        // This is a simplified version - in production use
        // SensorManager.getRotationMatrix()
        orientation[0] += gyroValues[0] * dt; // Roll
        orientation[1] += gyroValues[1] * dt; // Pitch
        orientation[2] += gyroValues[2] * dt; // Yaw

        gyroData = gyroValues.clone();
    }

    private float[] normalizeMagneticVector(float[] magneticVector) {
        // Apply rotation normalization as per paper equation (13)
        // This rotates the magnetic vector to a reference frame (e.g., north-aligned)
        float yaw = orientation[2]; // Current heading

        // Rotation matrix for yaw (rotation around Z-axis)
        float cosYaw = (float) Math.cos(yaw);
        float sinYaw = (float) Math.sin(yaw);

        // Rotate to align with north (reference direction = 0 degrees)
        float bx = magneticVector[0];
        float by = magneticVector[1];
        float bz = magneticVector[2];

        // Rotate around Z-axis by -yaw to align with north
        float bxRotated = bx * cosYaw + by * sinYaw;
        float byRotated = -bx * sinYaw + by * cosYaw;

        return new float[] { bxRotated, byRotated, bz };
    }

    private float calculateMagneticMagnitude(float[] magneticVector) {
        return (float) Math.sqrt(
                magneticVector[0] * magneticVector[0] +
                        magneticVector[1] * magneticVector[1] +
                        magneticVector[2] * magneticVector[2]);
    }

    public void startInitialPositionEstimation() {
        Log.d("MagneticLocalization", "Starting initial position estimation");
        isCollecting = true;
        stepCounter = 0;
        magneticSequence.clear();
    }

    public void onStepDetected() {
        if (isCollecting && stepCounter < REQUIRED_STEPS) {
            stepCounter++;
            Log.d("MagneticLocalization", "Step " + stepCounter + " detected for magnetic collection");

            if (stepCounter == REQUIRED_STEPS) {
                // We have collected enough steps, process the data
                processMagneticSequenceForInitialization();
            }
        }
    }

    private void processMagneticSequenceForInitialization() {
        if (magneticSequence.size() < REQUIRED_STEPS) {
            Log.e("MagneticLocalization", "Not enough magnetic data collected");
            return;
        }

        Log.d("MagneticLocalization", "Processing magnetic sequence for initialization");

        // Use the last magnetic reading (10th step) for position estimation
        float[] lastMagnetic = magneticSequence.get(magneticSequence.size() - 1);

        // Step 1: Estimate position using k-NN regression
        float[] estimatedPosition = estimatePositionWithKNN(lastMagnetic);

        if (estimatedPosition != null) {
            float x = estimatedPosition[0];
            float y = estimatedPosition[1];
            int floor = (int) estimatedPosition[2];

            Log.d("MagneticLocalization",
                    String.format("Initial position estimated: (%.2f, %.2f) on floor %d",
                            x, y, floor));

            if (positionListener != null) {
                positionListener.onInitialPositionEstimated(x, y, floor);
            }

            // Set this as current floor map for future calibrations
            currentFloorMap = findFloorMap(floor);
        }

        // Stop collecting
        isCollecting = false;
    }

    /**
     * Get current position estimate from magnetic reading.
     * Used for continuous particle filter updates.
     */
    public float[] getCurrentPosition(float[] currentMagnetic) {
        if (fingerprintDatabase == null || fingerprintDatabase.isEmpty()) {
            return null;
        }
        if (currentMagnetic == null || currentMagnetic.length < 3) {
            return null;
        }
        return estimatePositionWithKNN(currentMagnetic);
    }

    private float[] estimatePositionWithKNN(float[] queryMagnetic) {
        if (fingerprintDatabase.isEmpty()) {
            Log.e("MagneticLocalization", "Fingerprint database is empty");
            return null;
        }

        // Calculate distances to all fingerprints
        List<Neighbor> neighbors = new ArrayList<>();
        for (MagneticFingerprint fingerprint : fingerprintDatabase) {
            float distance = calculateEuclideanDistance(queryMagnetic, fingerprint.magneticVector);
            neighbors.add(new Neighbor(fingerprint, distance));
        }

        // Sort by distance
        neighbors.sort((n1, n2) -> Float.compare(n1.distance, n2.distance));

        // Take k nearest neighbors
        int k = Math.min(kValue, neighbors.size());
        List<Neighbor> kNearest = neighbors.subList(0, k);

        // Perform regression: average coordinates of k-nearest neighbors
        float sumX = 0, sumY = 0;
        int floor = kNearest.get(0).fingerprint.floor;

        for (Neighbor neighbor : kNearest) {
            sumX += neighbor.fingerprint.x;
            sumY += neighbor.fingerprint.y;
        }

        float avgX = sumX / kNearest.size();
        float avgY = sumY / kNearest.size();

        return new float[] { avgX, avgY, floor };
    }

    private float calculateEuclideanDistance(float[] v1, float[] v2) {
        // Euclidean distance as per paper equation (14)
        float sum = 0;
        for (int i = 0; i < 3; i++) {
            float diff = v1[i] - v2[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }

    // Method for calibration node matching (to be called during floor transitions)
    public void checkForCalibrationNode(float[] currentMagnetic) {
        if (currentFloorMap == null)
            return;

        for (CalibrationNode node : currentFloorMap.calibrationNodes) {
            float distance = calculateEuclideanDistance(currentMagnetic, node.magneticVector);

            // Threshold for matching (can be tuned)
            if (distance < 5.0f) { // μT threshold
                Log.d("MagneticLocalization",
                        String.format("Calibration node matched: %s at (%.2f, %.2f)",
                                node.transitionId, node.x, node.y));

                if (positionListener != null) {
                    positionListener.onCalibrationNodeMatched(node.x, node.y);
                }
                break;
            }
        }
    }

    private FloorMap findFloorMap(int floorNumber) {
        // In real implementation, you would have multiple floor maps
        // For simplicity, return a single floor map or create based on floor number
        FloorMap floorMap = new FloorMap(floorNumber, "Default Building");

        // Add some test calibration nodes
        floorMap.calibrationNodes.add(new CalibrationNode(
                10.0f, 5.0f, floorNumber,
                new float[] { 25.0f, -5.0f, 45.0f },
                "staircase", "S1"));

        return floorMap;
    }

    // Database management methods
    public void loadFingerprintDatabase(String buildingName) {
        // Load from assets or network
        // This would typically load pre-collected magnetic field maps
        fingerprintDatabase.clear();

        // Example: Load from JSON file in assets
        // For now, create dummy data for testing
        // DISABLED: createTestFingerprints() was generating 100 random fingerprints
        // that override the sample fingerprints loaded by SampleFingerprintLoader
        // createTestFingerprints();

        Log.d("MagneticLocalization", "Fingerprint database cleared, ready for sample data");
    }

    private void createTestFingerprints() {
        // DISABLED: This was creating 100 random fingerprints that override sample data
        // Create test fingerprints (in real app, load from file)
        /*
         * for (int i = 0; i < 100; i++) {
         * float x = (float) (Math.random() * 100); // 0-100 meters
         * float y = (float) (Math.random() * 50); // 0-50 meters
         * float bx = (float) (20 + Math.random() * 10); // Bx ~20-30 μT
         * float by = (float) (-10 + Math.random() * 20); // By ~-10 to 10 μT
         * float bz = (float) (40 + Math.random() * 20); // Bz ~40-60 μT
         * 
         * fingerprintDatabase.add(new MagneticFingerprint(
         * x, y, 1, new float[]{bx, by, bz}
         * ));
         * }
         * Log.d("MagneticLocalization", "Created " + fingerprintDatabase.size() +
         * " test fingerprints");
         */
    }

    public void setPositionEstimationListener(PositionEstimationListener listener) {
        this.positionListener = listener;
    }

    // Helper class for KNN
    private static class Neighbor {
        MagneticFingerprint fingerprint;
        float distance;

        Neighbor(MagneticFingerprint fingerprint, float distance) {
            this.fingerprint = fingerprint;
            this.distance = distance;
        }
    }
}