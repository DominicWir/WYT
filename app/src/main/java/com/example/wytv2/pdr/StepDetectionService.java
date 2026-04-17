package com.example.wytv2.pdr;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import com.example.wytv2.pdr.magneticfield.MagneticFieldLocalizationService;
import com.example.wytv2.SensorFeatureBuffer;
import com.example.wytv2.SimplifiedActivityModel;
import com.example.wytv2.localization.ParticleFilterLocalization;
import com.example.wytv2.localization.Position;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StepDetectionService implements SensorEventListener {
    private SensorManager sensorManager;
    private Sensor accelerometer;
    private Sensor magnetometer;
    private Sensor gyroscope;
    private Sensor stepDetectorSensor; // Android hardware step detector (primary step source)
    private Context context;

    // Step detection parameters
    private float binaryThreshold = 0.4f; // Lowered to catch genuine walking footfalls
    private int windowSize = 10; // Smaller window = faster detection (10 samples @ 60Hz = ~167ms)
    private final int MIN_STEP_INTERVAL_MS = 500; // 500ms = max 120 steps/min; PRIMARY duplicate guard

    // Acceleration data buffers
    private List<Float> rawAccelerationBuffer = new ArrayList<>();
    private List<Float> smoothedAccelerationBuffer = new ArrayList<>();
    private static final int ACCEL_BUFFER_SIZE = 100; // 2 seconds at 50Hz

    // Step tracking
    private int totalStepCount = 0;
    private int sessionStepCount = 0;
    private long lastStepTime = 0;
    private long lastStepDetectionTime = 0;

    // Movement bout tracking — distinguishes brief device motion from sustained
    // walking
    private long movementBoutStartTime = 0; // When did this movement bout begin?
    private static final long BURST_THRESHOLD_MS = 2000; // < 2s = device shake/short move
    private static final long BOUT_RESET_MS = 3000; // Reset bout if no step for 3s
    private static final float BURST_STEP_LENGTH = 0.3f; // Small displacement for short bursts
    private static final float MAX_WALKING_STEP = 0.8f; // Cap for sustained walking steps

    // Step cadence validation (filters hand shaking vs walking)
    // Walking is rhythmic; hand shaking is erratic.
    // Buffer step candidates and only count when cadence is consistent.
    private List<Long> stepCandidateTimestamps = new ArrayList<>();
    private boolean stepsValidated = false;
    private static final int STEP_VALIDATION_COUNT = 3; // Need 3 consistent steps (was 4)
    private static final float MAX_CADENCE_CV = 0.50f; // Max coefficient of variation (50%)
    private static final long STEP_CANDIDATE_TIMEOUT_MS = 5000; // Discard buffer after 5s gap (was 4s)

    // For step length calculation
    private float stepMaxAcc = 0;
    private float stepMinAcc = Float.MAX_VALUE;
    private List<Float> currentStepAccelerations = new ArrayList<>();

    // Stationary detection
    private boolean isStationary = false;
    private boolean wasStationary = false;
    private List<Float> stationaryBuffer = new ArrayList<>();
    private static final int STATIONARY_BUFFER_SIZE = 50; // ~0.8s at 60Hz (was 80 = ~1.3s)
    private static final float STATIONARY_THRESHOLD = 0.12f;
    /**
     * When false (e.g. edit mode open) skip all stationary position corrections.
     */
    private volatile boolean correctionEnabled = true;

    public void setPositionCorrectionEnabled(boolean enabled) {
        correctionEnabled = enabled;
    }

    // Sustained movement gate — brief jostles (<1.5s) don't count as walking
    private long movingStartTime = 0;
    private static final long MIN_MOVING_DURATION_MS = 1500; // Must be moving for 1.5s before steps count

    // ---- Sensor-Fingerprint Anchor System ----
    // Anchors store WiFi RSSI + magnetic magnitude snapshots taken when stationary.
    // Matching is done in SENSOR space (not drift-prone PDR space), so only genuine
    // physical revisits of a location trigger position correction.
    private static class SensorAnchor {
        final Map<String, Integer> wifiRssi; // BSSID → RSSI fingerprint
        final float magMagnitude; // |B| at this location (µT)
        final float[] pdrPosition; // PDR [x, y] when anchor was saved

        SensorAnchor(Map<String, Integer> wifi, float mag, float x, float y) {
            this.wifiRssi = wifi;
            this.magMagnitude = mag;
            this.pdrPosition = new float[] { x, y };
        }
    }

    private final List<SensorAnchor> sensorAnchors = new ArrayList<>();
    private static final int MAX_ANCHORS = 30;
    // Sensor similarity thresholds for anchor matching
    private static final float ANCHOR_WIFI_SIM_THRESHOLD = 0.80f; // cosine similarity (0-1)
    private static final float ANCHOR_MAG_DIFF_THRESHOLD = 5.0f; // µT tolerance
    // Minimum physical separation: only save a new anchor if far enough from all
    // existing ones
    private static final float ANCHOR_PDR_MIN_SEPARATION = 1.0f; // metres

    // Zone positions for zone-constrained anchor correction.
    // Populated by MainActivity whenever zones are saved/deleted.
    private final List<float[]> zonePositions = new ArrayList<>(); // [x, y] per zone

    /** Called whenever zone markers change. Allows zone-constrained correction. */
    public synchronized void setZonePositions(List<float[]> positions) {
        zonePositions.clear();
        zonePositions.addAll(positions);
    }

    /**
     * Returns the index of the nearest zone position to (x, y), or -1 if no zones
     * defined.
     */
    private int nearestZoneIndex(float x, float y) {
        if (zonePositions.isEmpty())
            return -1;
        int best = 0;
        float bestDist2 = Float.MAX_VALUE;
        for (int i = 0; i < zonePositions.size(); i++) {
            float dx = x - zonePositions.get(i)[0];
            float dy = y - zonePositions.get(i)[1];
            float d2 = dx * dx + dy * dy;
            if (d2 < bestDist2) {
                bestDist2 = d2;
                best = i;
            }
        }
        return best;
    }

    /**
     * Register a zone creation point as a SensorAnchor immediately using current
     * WiFi and magnetic snapshots. Called by MainActivity when user saves a zone.
     */
    public void addZoneAnchor(float x, float y) {
        List<com.example.wytv2.wifi.WiFiReading> wifiReadings = (wifiService != null) ? wifiService.getCurrentReadings()
                : null;
        Map<String, Integer> wifi = new HashMap<>();
        if (wifiReadings != null) {
            for (com.example.wytv2.wifi.WiFiReading r : wifiReadings)
                wifi.put(r.bssid, r.rssi);
        }
        float mag = (latestMag != null) ? (float) Math.sqrt(
                latestMag[0] * latestMag[0] + latestMag[1] * latestMag[1] + latestMag[2] * latestMag[2]) : 0f;
        sensorAnchors.add(new SensorAnchor(wifi, mag, x, y));
        if (sensorAnchors.size() > MAX_ANCHORS)
            sensorAnchors.remove(0);
        Log.d("StepDetection", String.format(
                "Zone anchor added at (%.2f,%.2f) mag=%.1fµT wifi=%d APs [%d total]",
                x, y, mag, wifi.size(), sensorAnchors.size()));
    }

    /**
     * Reset the particle filter to the coordinate origin (0, 0).
     * Called when switching to a new floor map — each floor has its own coordinate
     * space.
     */
    public void resetToOrigin() {
        if (particleFilter != null && particleFilter.isInitialized()) {
            particleFilter.initialize(new Position(0, 0, 0), 500, currentHeading, 0.5f);
            sensorAnchors.clear();
            Log.d("StepDetection", "Particle filter reset to origin for new floor map");
        }
    }

    /**
     * Reset the particle filter to a known position (e.g. last zone before a floor
     * transition).
     * Lower spread (0.3f) since the position is authoritative.
     */
    public void resetToPosition(float x, float y) {
        if (particleFilter != null && particleFilter.isInitialized()) {
            particleFilter.initialize(new Position(x, y, 0), 500, currentHeading, 0.3f);
            sensorAnchors.clear();
            Log.d("StepDetection", String.format(
                    "Particle filter reset to known position (%.2f, %.2f)", x, y));
        }
    }

    private float latestAcceleration = 0; // used by getLatestAcceleration()

    // ---- Non-walking motion detector ----
    // Rapid direction changes (high jerk) = device shaking, not walking
    // Walking: ~2 sign reversals/sec; Shaking: 6+ per sec
    private float lastLinearAcc = 0; // Previous linearAcc derivative (for jerk tracking)
    private List<Long> signChangeTimestamps = new ArrayList<>(); // Times of sign changes
    private static final float JERK_DEAD_BAND = 0.08f; // Ignore tiny noise flips below this
    private static final int SHAKE_CHANGE_THRESHOLD = 8; // > 8 reversals in 1s = not walking (was 5, noise-prone)
    private static final long SHAKE_WINDOW_MS = 1000; // 1-second detection window
    private boolean nonWalkingMotionBlocked = false; // Blocks step detection when true

    // ---- Per-axis gravity tracking for swing detection ----
    private float gravX = 0, gravY = 0, gravZ = 0; // Low-pass filtered gravity per axis
    // α=0.95 → τ ≈ 330ms at 60Hz: correctly separates gravity from 1Hz arm swing
    // α=0.80 → τ ≈ 75ms at 60Hz: gravity absorbs arm swing (too fast — was the bug)
    private static final float GRAVITY_ALPHA = 0.95f;

    // ---- Pendulum swing step detector ----
    // Arm/leg swing during walking creates a sinusoidal oscillation on the dominant
    // axis.
    // One peak of this oscillation = one step.
    private static final int SWING_BUF_CAPACITY = 30; // ~500ms of samples for variance computation
    private java.util.ArrayDeque<float[]> swingAxisBuffer = new java.util.ArrayDeque<>();
    private int dominantAxis = 1; // 0=X, 1=Y, 2=Z; updated dynamically
    private long lastDomAxisUpdate = 0;
    private static final long DOM_AXIS_UPDATE_MS = 500; // Recompute dominant axis every 500ms
    private float prevDomProj = 0; // Previous sample on dominant axis
    private float prevPrevDomProj = 0; // Two samples back
    private static final float SWING_PEAK_THRESHOLD = 0.3f; // Min peak amplitude to be a step (m/s²)
    // Magnetic field localization
    private MagneticFieldLocalizationService magneticService;
    private boolean isInitializingPosition = false;
    private boolean hasInitialPosition = false;
    private float[] initialPosition = null; // [x, y, floor]
    private static final int INITIALIZATION_STEPS = 10; // First 10 steps for initialization

    // Calibration node tracking
    private List<CalibrationNode> matchedCalibrationNodes = new ArrayList<>();

    // ML Model Integration
    private SensorFeatureBuffer featureBuffer = new SensorFeatureBuffer();
    private long lastActivityPredictionTime = 0;
    private static final long ACTIVITY_PREDICTION_INTERVAL_MS = 2000; // Every 2 seconds
    private String currentActivity = "Unknown";
    private float currentActivityConfidence = 0.0f;

    // Latest sensor values for feature extraction
    private float[] latestAccel = new float[3];
    private float[] latestGyro = new float[3];
    private float[] latestMag = new float[3];

    // Heading tracking for particle filter
    private float currentHeading = 0.0f;
    private float previousStepHeading = 0.0f;
    private float compassHeading = 0.0f; // Magnetometer-based heading
    private static final float HEADING_FILTER_ALPHA = 0.90f; // 90% gyro, 10% compass (was 0.98, too much drift)
    private long lastGyroTime = 0; // Use System.nanoTime() instead of sensor timestamp

    // WiFi RSSI Integration
    private com.example.wytv2.wifi.WiFiRSSIService wifiService;
    private com.example.wytv2.wifi.WiFiFingerprintDatabase wifiDatabase;
    private float wifiRssiVariance = 0.0f;
    private static final float RSSI_STABLE_THRESHOLD = 5.0f; // Low variance = stationary

    // Particle Filter for Sensor Fusion
    private com.example.wytv2.localization.ParticleFilterLocalization particleFilter;
    private com.example.wytv2.localization.Position fusedPosition;

    // Stationary drift prevention
    private long stationaryStartTime = System.currentTimeMillis(); // Init to now so first onStartedMoving() doesn't see
                                                                   // 29M-min duration
    private com.example.wytv2.localization.Position lastPositionBeforeStationary = null;
    private static final long STATIONARY_RESET_DELAY_MS = 10000; // 10 seconds
    private android.os.Handler driftPreventionHandler = new android.os.Handler();
    private boolean driftPreventionResetPerformed = false; // Track if reset already done

    // Continuous Learning
    private com.example.wytv2.ml.StepDataCollector dataCollector;
    private com.example.wytv2.ml.RetrainingScheduler retrainingScheduler;

    // ML WiFi Positioning Model
    private com.example.wytv2.ml.WiFiPositioningModel wifiPositioningModel;

    // Callback
    public interface DeviceStateListener {
        void onDeviceStateChanged(boolean isStationary, float confidence);

        void onStepDetected(long timestamp, float stepLength, int totalSteps, int sessionSteps);

        void onStepsReset();

        void onInitialPositionEstimated(float x, float y, int floor);

        void onCalibrationNodeMatched(float x, float y, String nodeId);

        void onMagneticDataCollected(float[] magneticVector);

        void onActivityRecognized(String activity, float confidence);

        void onWiFiRSSIUpdate(java.util.List<com.example.wytv2.wifi.WiFiReading> readings, float variance);

        void onPositionUpdated(com.example.wytv2.localization.Position fusedPosition);

        /** Called when the ML WiFi positioning model produces a prediction. */
        default void onMLPositionPrediction(float x, float y, int floor) {
            // Default no-op — override to use ML position predictions
        }
    }

    /**
     * Callback interface for calibration mode.
     */
    public interface CalibrationListener {
        /**
         * Called when a step is detected during calibration.
         * 
         * @param stepCount Current step count
         * @param threshold Current detection threshold
         */
        void onCalibrationStepDetected(int stepCount, float threshold);
    }

    // List of listeners
    private java.util.List<DeviceStateListener> listeners = new java.util.ArrayList<>();
    private CalibrationListener calibrationListener;

    // Calibration mode
    private boolean calibrationMode = false;

    // Calibration Node class
    public static class CalibrationNode {
        public float x, y;
        public int floor;
        public float[] magneticVector;
        public String type; // "staircase", "elevator", "landmark"
        public String nodeId; // e.g., "S1", "E1"

        public CalibrationNode(float x, float y, int floor, float[] magneticVector,
                String type, String nodeId) {
            this.x = x;
            this.y = y;
            this.floor = floor;
            this.magneticVector = magneticVector.clone();
            this.type = type;
            this.nodeId = nodeId;
        }
    }

    // Binder for service binding
    public class LocalBinder extends android.os.Binder {
        public StepDetectionService getService() {
            return StepDetectionService.this;
        }
    }

    private final android.os.IBinder binder = new LocalBinder();

    public android.os.IBinder getBinder() {
        return binder;
    }

    public StepDetectionService(Context context) {
        this.context = context;
        sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        // Hardware step detector (primary)
        stepDetectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR);

        // Accelerometer (gravity/heading/stationary)
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        Log.d("StepDetection", "=== StepDetectionService constructor START ===");

        // Initialize WiFi RSSI service
        wifiService = new com.example.wytv2.wifi.WiFiRSSIService(context);
        wifiDatabase = new com.example.wytv2.wifi.WiFiFingerprintDatabase();

        // Initialize Particle Filter
        particleFilter = new com.example.wytv2.localization.ParticleFilterLocalization();

        // QUICK FIX: Initialize immediately with default position and current heading
        if (particleFilter != null) {
            com.example.wytv2.localization.Position defaultPos = new com.example.wytv2.localization.Position(0.0f, 0.0f,
                    1, 0.5f);
            // Use current heading from gyroscope (or 0 if not available yet)
            particleFilter.initialize(defaultPos, 500, currentHeading);
            hasInitialPosition = true;
            initialPosition = new float[] { 0.0f, 0.0f, 1.0f };
            Log.d("StepDetection", String.format("✓ Particle filter initialized at (0, 0) with heading %.1f°",
                    Math.toDegrees(currentHeading)));
        }

        // Wrap ML/optional init in try-catch so it can't crash step detection
        try {
            // Initialize Activity Model with persistence support
            com.example.wytv2.SimplifiedActivityModel.initialize(context);

            // Initialize Continuous Learning Data Collector
            dataCollector = new com.example.wytv2.ml.StepDataCollector(context);

            // Initialize Model Retrainer and load saved threshold
            com.example.wytv2.ml.ModelRetrainer modelRetrainer = new com.example.wytv2.ml.ModelRetrainer(context,
                    dataCollector);

            // Load saved threshold from previous sessions (continuous learning!)
            float savedThreshold = modelRetrainer.getCurrentThreshold();
            this.binaryThreshold = savedThreshold;
            Log.d("StepDetection", String.format(
                    "✓ Loaded saved threshold: %.2f (accuracy: %.1f%%)",
                    savedThreshold, modelRetrainer.getCurrentAccuracy() * 100));

            // Set up listener to update threshold when retraining completes
            modelRetrainer
                    .setThresholdUpdateListener(new com.example.wytv2.ml.ModelRetrainer.ThresholdUpdateListener() {
                        @Override
                        public void onThresholdUpdated(float newThreshold, String reason) {
                            binaryThreshold = newThreshold;
                            Log.i("StepDetection", "Threshold updated: " + reason);
                        }
                    });

            // Initialize Retraining Scheduler (prototype settings: every 50 steps)
            retrainingScheduler = new com.example.wytv2.ml.RetrainingScheduler(context, dataCollector);
            retrainingScheduler.setRetrainingInterval(50); // Frequent for prototyping
            retrainingScheduler.setMinSamplesRequired(30);
        } catch (Exception e) {
            Log.e("StepDetection", "ML/retrainer init failed (non-fatal): " + e.getMessage());
        }

        // Initialize ML WiFi Positioning Model ASYNC to avoid blocking main thread
        // The model is ~6MB and ONNX Runtime initialization can be slow
        wifiPositioningModel = new com.example.wytv2.ml.WiFiPositioningModel();
        new Thread(() -> {
            try {
                boolean mlModelLoaded = wifiPositioningModel.initialize(context);
                Log.d("StepDetection", mlModelLoaded
                        ? "✓ WiFi positioning ML model loaded (async)"
                        : "⚠ WiFi positioning ML model not available (using k-NN fallback)");
            } catch (Exception e) {
                Log.e("StepDetection", "ML model init failed (non-fatal): " + e.getMessage());
            }
        }, "MLModelInit").start();

        // Set WiFi listener for false positive detection
        wifiService.setRSSIListener(new com.example.wytv2.wifi.WiFiRSSIService.RSSIListener() {
            @Override
            public void onRSSIUpdate(java.util.List<com.example.wytv2.wifi.WiFiReading> readings, float variance) {
                wifiRssiVariance = variance;
                checkFalsePositiveWithWiFi();

                // Forward to UI listener
                for (DeviceStateListener listener : listeners) {
                    listener.onWiFiRSSIUpdate(readings, variance);
                }
            }

            @Override
            public void onScanComplete(java.util.List<com.example.wytv2.wifi.WiFiReading> readings) {
                // Feed WiFi RSSI values to ML positioning model
                if (wifiPositioningModel != null && readings != null && !readings.isEmpty()) {
                    float[] rssiValues = new float[readings.size()];
                    for (int i = 0; i < readings.size(); i++) {
                        rssiValues[i] = readings.get(i).rssi;
                    }
                    wifiPositioningModel.updateWifiReadings(rssiValues, readings.size());
                    Log.d("WiFiML", String.format("Updated ML model with %d WiFi APs",
                            readings.size()));
                }

                // Also run k-NN estimation (currently disabled)
                if (wifiDatabase != null && readings != null && readings.size() >= 3) {
                    com.example.wytv2.wifi.WiFiFingerprintDatabase.Position wifiEstimate = wifiDatabase
                            .estimatePosition(readings, 3);

                    if (wifiEstimate != null && particleFilter != null &&
                            particleFilter.isInitialized()) {
                        // k-NN WiFi positioning disabled for now — ML model handles this
                        Log.d("WiFiML", String.format(
                                "k-NN estimate: (%.2f, %.2f) — ML model active for positioning",
                                wifiEstimate.x, wifiEstimate.y));
                    }
                }
            }
        });

        // Load calibrated threshold if available
        loadCalibratedThreshold();

        // Initialize magnetic field localization service
        magneticService = new MagneticFieldLocalizationService(context);
        magneticService.setPositionEstimationListener(
                new MagneticFieldLocalizationService.PositionEstimationListener() {
                    @Override
                    public void onInitialPositionEstimated(float x, float y, int floor) {
                        Log.d("StepDetection",
                                String.format("✓ Initial position from magnetic: (%.2f, %.2f) floor %d",
                                        x, y, floor));

                        hasInitialPosition = true;
                        initialPosition = new float[] { x, y, floor };

                        // DO NOT re-initialize particle filter - it's already initialized at (0,0)
                        // The magnetic reading may be inaccurate and cause large position offset
                        // Instead, let the filter correct itself through updateWithMagnetic()
                        /*
                         * if (particleFilter != null) {
                         * com.example.wytv2.localization.Position initPos = new
                         * com.example.wytv2.localization.Position(
                         * x, y, floor, 0.8f);
                         * particleFilter.initialize(initPos, 500);
                         * 
                         * Log.d("StepDetection", String.format(
                         * "Particle filter initialized at (%.1f, %.1f) with %d particles",
                         * x, y, particleFilter.getParticleCount()));
                         * }
                         */

                        for (DeviceStateListener listener : listeners) {
                            listener.onInitialPositionEstimated(x, y, floor);
                        }

                        // Set the initial position for PDR tracking
                        // This would be used if we were maintaining a position estimate
                    }

                    @Override
                    public void onCalibrationNodeMatched(float x, float y) {
                        Log.d("StepDetection",
                                String.format("✓ Calibration node at (%.2f, %.2f)", x, y));

                        // Store matched node
                        matchedCalibrationNodes.add(new CalibrationNode(
                                x, y,
                                initialPosition != null ? (int) initialPosition[2] : 1,
                                new float[] { 0, 0, 0 }, // Would use actual magnetic data
                                "staircase",
                                "CN" + matchedCalibrationNodes.size()));

                        for (DeviceStateListener listener : listeners) {
                            listener.onCalibrationNodeMatched(x, y,
                                    "CN" + matchedCalibrationNodes.size());
                        }

                        // In a full implementation, you would correct your position here
                    }

                    @Override
                    public void onMagneticDataCollected(float[] magneticVector) {
                        for (DeviceStateListener listener : listeners) {
                            listener.onMagneticDataCollected(magneticVector);
                        }
                    }
                });

        // Load magnetic fingerprint database (in real app, load from assets)
        magneticService.loadFingerprintDatabase("TestBuilding");
        // Load sample fingerprints for testing
        SampleFingerprintLoader.loadSamples(wifiDatabase, magneticService);
    }

    public void addDeviceStateListener(DeviceStateListener listener) {
        if (!listeners.contains(listener)) {
            listeners.add(listener);
            Log.d("StepDetection", String.format(
                    "✓ Listener added: %s (total listeners: %d)",
                    listener.getClass().getSimpleName(), listeners.size()));
        } else {
            Log.w("StepDetection", String.format(
                    "⚠ Listener already registered: %s",
                    listener.getClass().getSimpleName()));
        }
    }

    public void removeDeviceStateListener(DeviceStateListener listener) {
        boolean removed = listeners.remove(listener);
        if (removed) {
            Log.d("StepDetection", String.format(
                    "✓ Listener removed: %s (remaining: %d)",
                    listener.getClass().getSimpleName(), listeners.size()));
        } else {
            Log.w("StepDetection", String.format(
                    "⚠ Listener not found for removal: %s",
                    listener.getClass().getSimpleName()));
        }
    }

    public void setCalibrationListener(CalibrationListener listener) {
        this.calibrationListener = listener;
    }

    public void setThreshold(float threshold) {
        this.binaryThreshold = threshold;
        Log.d("StepDetection", "Threshold updated to: " + threshold);
    }

    /**
     * Start calibration mode.
     * Resets step counter and enables calibration callbacks.
     */
    public void startCalibration(CalibrationListener listener) {
        this.calibrationMode = true;
        this.calibrationListener = listener;
        this.sessionStepCount = 0;
        Log.d("StepDetection", "Calibration mode started");
    }

    /**
     * Stop calibration mode.
     * Returns the number of steps detected during calibration.
     */
    public int stopCalibration() {
        this.calibrationMode = false;
        int detectedSteps = this.sessionStepCount;
        this.calibrationListener = null;
        Log.d("StepDetection", "Calibration mode stopped. Detected: " + detectedSteps);
        return detectedSteps;
    }

    /**
     * Save calibrated threshold to SharedPreferences.
     */
    public void saveCalibratedThreshold(float threshold) {
        android.content.SharedPreferences prefs = context.getSharedPreferences("StepDetection", Context.MODE_PRIVATE);
        prefs.edit()
                .putFloat("calibrated_threshold", threshold)
                .putBoolean("is_calibrated", true)
                .apply();

        this.binaryThreshold = threshold;
        Log.d("StepDetection", "Calibrated threshold saved: " + threshold);
    }

    /**
     * Load calibrated threshold from SharedPreferences.
     */
    private void loadCalibratedThreshold() {
        android.content.SharedPreferences prefs = context.getSharedPreferences("StepDetection", Context.MODE_PRIVATE);

        if (prefs.getBoolean("is_calibrated", false)) {
            float calibratedThreshold = prefs.getFloat("calibrated_threshold", 1.0f);
            this.binaryThreshold = calibratedThreshold;
            Log.d("StepDetection", "Loaded calibrated threshold: " + calibratedThreshold);
        }
    }

    public void start() {
        // Primary: hardware step detector (manufacturer-tuned, filters non-walking
        // motion)
        if (stepDetectorSensor != null) {
            sensorManager.registerListener(this, stepDetectorSensor, SensorManager.SENSOR_DELAY_NORMAL);
            Log.d("StepDetection", "Hardware step detector registered (primary step source)");
        } else {
            Log.w("StepDetection", "Hardware step detector unavailable — falling back to swing detection");
        }

        // Always register accelerometer for heading, stationary, and fallback swing
        // detection
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_UI);
        if (magnetometer != null) {
            sensorManager.registerListener(this, magnetometer, SensorManager.SENSOR_DELAY_UI);
        }
        if (gyroscope != null) {
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_UI);
        }

        // Start WiFi RSSI scanning
        if (wifiService != null) {
            wifiService.startScanning();
        }

        Log.d("StepDetection", "Service started");
    }

    public void stop() {
        sensorManager.unregisterListener(this);

        // Stop WiFi scanning
        if (wifiService != null) {
            wifiService.stopScanning();
        }

        // Stop drift prevention checks
        if (driftPreventionHandler != null) {
            driftPreventionHandler.removeCallbacks(this::checkAndPreventDrift);
        }

        // Close ML positioning model
        if (wifiPositioningModel != null) {
            wifiPositioningModel.close();
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        long currentTime = System.currentTimeMillis();

        switch (event.sensor.getType()) {
            case Sensor.TYPE_STEP_DETECTOR:
                // Hardware step detector fired — a real walking step was detected
                // This is the primary step source when hardware sensor is available
                Log.d("StepDetection", "Hardware step detected");
                onStepDetected(currentTime);
                lastStepTime = currentTime;
                lastStepDetectionTime = currentTime;
                resetStepTracking();
                break;

            case Sensor.TYPE_ACCELEROMETER:
                processAccelerometer(event.values, currentTime);
                break;

            case Sensor.TYPE_MAGNETIC_FIELD:
                processMagnetometer(event.values, currentTime);
                break;

            case Sensor.TYPE_GYROSCOPE:
                processGyroscope(event.values, currentTime);
                break;
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Not needed
    }

    private void processAccelerometer(float[] values, long timestamp) {
        // Store for feature extraction
        latestAccel[0] = values[0];
        latestAccel[1] = values[1];
        latestAccel[2] = values[2];

        float x = values[0];
        float y = values[1];
        float z = values[2];

        // Per-axis gravity removal via low-pass filter
        gravX = GRAVITY_ALPHA * gravX + (1 - GRAVITY_ALPHA) * x;
        gravY = GRAVITY_ALPHA * gravY + (1 - GRAVITY_ALPHA) * y;
        gravZ = GRAVITY_ALPHA * gravZ + (1 - GRAVITY_ALPHA) * z;
        float linX = x - gravX;
        float linY = y - gravY;
        float linZ = z - gravZ;

        // Magnitude for stationary detection
        float magnitude = (float) Math.sqrt(x * x + y * y + z * z);
        float linearAcc = removeGravity(magnitude);
        latestAcceleration = linearAcc;

        // Update stationary detection (for UI + drift prevention)
        wasStationary = isStationary;
        updateStationaryDetection(linearAcc);
        if (wasStationary && !isStationary) {
            onStartedMoving();
        }

        // Block step detection when device is genuinely stationary
        if (isStationary) {
            movingStartTime = 0;
            return;
        }

        // If hardware step detector is available, skip custom swing detection
        // (hardware sensor is already handling steps via onSensorChanged
        // TYPE_STEP_DETECTOR)
        if (stepDetectorSensor != null)
            return;

        // Fallback: swing-based step detection (only runs if no hardware step detector)
        if (movingStartTime == 0)
            movingStartTime = timestamp;
        if (timestamp - movingStartTime < MIN_MOVING_DURATION_MS)
            return;
        detectSwingStep(linX, linY, linZ, linearAcc, timestamp);
    }

    /**
     * Pendulum swing step detector.
     * Walking (hand or pocket) creates sinusoidal oscillation on the dominant axis.
     * Each local maximum of the projection on that axis = one step.
     * Uses gravity vector orientation to exclude the vertical axis from swing
     * detection.
     */
    private void detectSwingStep(float linX, float linY, float linZ, float linearAcc, long timestamp) {
        // Accumulate samples for variance computation
        swingAxisBuffer.offer(new float[] { linX, linY, linZ });
        if (swingAxisBuffer.size() > SWING_BUF_CAPACITY) {
            swingAxisBuffer.poll();
        }

        // Recompute dominant swing axis every 500ms
        if (swingAxisBuffer.size() >= 10 && (timestamp - lastDomAxisUpdate) > DOM_AXIS_UPDATE_MS) {
            // Step 1: Find the VERTICAL axis using gravity vector
            // The axis with highest |gravity| component = vertical (mostly gravity, no
            // swing signal)
            float absGX = Math.abs(gravX), absGY = Math.abs(gravY), absGZ = Math.abs(gravZ);
            int verticalAxis = 0;
            float maxGrav = absGX;
            if (absGY > maxGrav) {
                maxGrav = absGY;
                verticalAxis = 1;
            }
            if (absGZ > maxGrav) {
                verticalAxis = 2;
            }

            // Step 2: The two remaining (horizontal) axes are candidates for swing
            // detection
            int h1 = (verticalAxis + 1) % 3;
            int h2 = (verticalAxis + 2) % 3;

            // Step 3: Pick the horizontal axis with higher variance = the swing axis
            float[] mean = { 0, 0, 0 };
            for (float[] s : swingAxisBuffer) {
                mean[0] += s[0];
                mean[1] += s[1];
                mean[2] += s[2];
            }
            int n = swingAxisBuffer.size();
            mean[0] /= n;
            mean[1] /= n;
            mean[2] /= n;

            float[] var = { 0, 0, 0 };
            for (float[] s : swingAxisBuffer) {
                var[0] += (s[0] - mean[0]) * (s[0] - mean[0]);
                var[1] += (s[1] - mean[1]) * (s[1] - mean[1]);
                var[2] += (s[2] - mean[2]) * (s[2] - mean[2]);
            }

            dominantAxis = (var[h1] >= var[h2]) ? h1 : h2;
            lastDomAxisUpdate = timestamp;

            // Diagnostic log every ~5s
            if (Math.random() < 0.003) {
                Log.d("StepDetection", String.format(
                        "Orientation: grav=[%.1f,%.1f,%.1f] vertAxis=%d swingAxis=%d var=[%.3f,%.3f,%.3f]",
                        gravX, gravY, gravZ, verticalAxis, dominantAxis,
                        var[0] / n, var[1] / n, var[2] / n));
            }
        }

        // Project linear acceleration onto the swing axis
        float proj = (dominantAxis == 0) ? linX : (dominantAxis == 1) ? linY : linZ;

        // Detect local maximum: prevPrev < prev AND prev > current AND prev > threshold
        // Peak of forward arm/leg swing = one step
        boolean isPeak = prevDomProj > prevPrevDomProj
                && prevDomProj > proj
                && prevDomProj > SWING_PEAK_THRESHOLD;

        prevPrevDomProj = prevDomProj;
        prevDomProj = proj;

        if (isPeak && (timestamp - lastStepTime) >= MIN_STEP_INTERVAL_MS) {
            Log.d("StepDetection", String.format(
                    "✓ Swing step | axis=%d peak=%.3f", dominantAxis, prevPrevDomProj));
            onStepDetected(timestamp);
            lastStepTime = timestamp;
            lastStepDetectionTime = timestamp;
            resetStepTracking();
        }
    }

    private void processMagnetometer(float[] values, long timestamp) {
        // Store for feature extraction
        latestMag[0] = values[0];
        latestMag[1] = values[1];
        latestMag[2] = values[2];

        // Calculate compass heading from magnetometer + accelerometer
        updateCompassHeading();

        // Pass magnetic data to the magnetic service
        magneticService.onMagneticData(values, timestamp);

        // Check for calibration nodes if we're moving
        if (!isStationary) {
            magneticService.checkForCalibrationNode(values);
        }
    }

    /**
     * Calculate compass heading from magnetometer and accelerometer data.
     * Uses Android's rotation matrix to get device orientation.
     */
    private void updateCompassHeading() {
        if (latestAccel == null || latestMag == null) {
            return;
        }

        float[] rotationMatrix = new float[9];
        float[] orientationAngles = new float[3];

        // Get rotation matrix from accelerometer and magnetometer
        boolean success = android.hardware.SensorManager.getRotationMatrix(
                rotationMatrix, null, latestAccel, latestMag);

        if (success) {
            // Get orientation angles (azimuth, pitch, roll)
            android.hardware.SensorManager.getOrientation(rotationMatrix, orientationAngles);

            // Azimuth (orientationAngles[0]) is the compass heading
            // Range: [-π, π] where 0 is North, π/2 is East, -π/2 is West
            compassHeading = orientationAngles[0];

            // Normalize to [0, 2π] to match gyroscope heading
            if (compassHeading < 0) {
                compassHeading += 2 * Math.PI;
            }

            // Debug logging (occasional)
            if (Math.random() < 0.05) {
                Log.d("Compass", String.format(
                        "Compass heading: %.1f° (raw azimuth: %.1f°)",
                        Math.toDegrees(compassHeading),
                        Math.toDegrees(orientationAngles[0])));
            }
        }
    }

    /**
     * Process gyroscope data to estimate heading.
     */
    private void processGyroscope(float[] values, long timestamp) {
        // Store for feature extraction
        latestGyro[0] = values[0];
        latestGyro[1] = values[1];
        latestGyro[2] = values[2];

        // Integrate angular velocity to get heading
        if (lastGyroTime != 0) {
            long currentTime = System.nanoTime();
            float dt = (currentTime - lastGyroTime) / 1_000_000_000.0f; // Convert to seconds

            // Use Z-axis (yaw) for heading in horizontal plane
            float angularVelocityZ = values[2]; // rad/s

            // Integrate to get heading change (gyroscope)
            float headingChange = angularVelocityZ * dt;
            float gyroHeading = currentHeading + headingChange;

            // Normalize gyro heading to [0, 2π]
            gyroHeading = (float) ((gyroHeading + 2 * Math.PI) % (2 * Math.PI));

            // Apply complementary filter: blend gyro (short-term) with compass (long-term)
            if (compassHeading > 0) { // Only if compass data available
                // Handle wraparound (e.g., gyro=359°, compass=1°)
                float diff = compassHeading - gyroHeading;
                if (diff > Math.PI)
                    diff -= 2 * Math.PI;
                if (diff < -Math.PI)
                    diff += 2 * Math.PI;

                // Blend: 98% gyro + 2% compass
                currentHeading = gyroHeading + (1 - HEADING_FILTER_ALPHA) * diff;
            } else {
                // No compass data yet, use gyro only
                currentHeading = gyroHeading;
            }

            // Normalize final heading to [0, 2π]
            currentHeading = (float) ((currentHeading + 2 * Math.PI) % (2 * Math.PI));

            // Debug logging every 10th sample to avoid spam
            if (Math.random() < 0.1) {
                Log.d("Gyroscope", String.format(
                        "AngVel: %.3f rad/s, dt: %.6fs, change: %.3f°, total: %.1f°",
                        angularVelocityZ, dt, Math.toDegrees(headingChange), Math.toDegrees(currentHeading)));
            }
        }

        lastGyroTime = System.nanoTime();

        // Gyroscope data can be used for orientation estimation
        // This would be used for magnetic vector rotation normalization
        if (!isStationary) {
            magneticService.updateGyroscopeData(values, timestamp);
        }

        // Add sensor data to feature buffer for ML inference
        addToFeatureBuffer(timestamp);
    }

    // High-pass filter for gravity removal
    private float gravity = 9.8f;

    private float removeGravity(float magnitude) {
        float alpha = 0.8f;
        gravity = alpha * gravity + (1 - alpha) * magnitude;
        return Math.abs(magnitude - gravity); // Use absolute value
    }

    private void processAccelerationForStep(float acceleration, long timestamp) {
        // Add to raw buffer
        rawAccelerationBuffer.add(acceleration);
        if (rawAccelerationBuffer.size() > ACCEL_BUFFER_SIZE) {
            rawAccelerationBuffer.remove(0);
        }

        // Apply smoothing (mean filter)
        float smoothed = smoothAcceleration(rawAccelerationBuffer);
        smoothedAccelerationBuffer.add(smoothed);
        if (smoothedAccelerationBuffer.size() > ACCEL_BUFFER_SIZE) {
            smoothedAccelerationBuffer.remove(0);
        }

        // Track min/max for step length calculation
        if (acceleration > stepMaxAcc)
            stepMaxAcc = acceleration;
        if (acceleration < stepMinAcc)
            stepMinAcc = acceleration;
        currentStepAccelerations.add(acceleration);

        // Check if we have enough data for step detection
        if (smoothedAccelerationBuffer.size() >= windowSize) {
            // Use sliding window approach
            detectStepWithSlidingWindow(timestamp);
        }

        // If it's been too long since last step, reset step tracking
        if (timestamp - lastStepDetectionTime > 2000) { // 2 seconds
            resetStepTracking();
        }
    }

    private float smoothAcceleration(List<Float> buffer) {
        if (buffer.isEmpty())
            return 0;

        // Simple moving average of last 5 samples
        int count = Math.min(5, buffer.size());
        float sum = 0;
        for (int i = buffer.size() - count; i < buffer.size(); i++) {
            sum += buffer.get(i);
        }
        return sum / count;
    }

    private void detectStepWithSlidingWindow(long timestamp) {
        // Get the latest window of smoothed data
        int startIdx = smoothedAccelerationBuffer.size() - windowSize;
        List<Float> window = new ArrayList<>();
        for (int i = startIdx; i < smoothedAccelerationBuffer.size(); i++) {
            window.add(smoothedAccelerationBuffer.get(i));
        }

        // Binarize the window
        int onesCount = 0;
        boolean[] binary = new boolean[windowSize];
        for (int i = 0; i < windowSize; i++) {
            binary[i] = window.get(i) > binaryThreshold;
            if (binary[i])
                onesCount++;
        }

        // A step peak: some samples above threshold (not too few = noise, not all =
        // sustained)
        // Primary duplicate guard is MIN_STEP_INTERVAL, not this window check
        if (onesCount >= 2 && onesCount <= windowSize - 2) {
            if (hasValidPulsePattern(binary)) {
                if (timestamp - lastStepTime >= MIN_STEP_INTERVAL_MS) {
                    onStepDetected(timestamp);
                    lastStepTime = timestamp;
                    lastStepDetectionTime = timestamp;
                    resetStepTracking();
                }
            }
        }
    }

    private boolean hasValidPulsePattern(boolean[] binary) {
        // Look for pattern: 1s followed by 0s (a pulse)
        boolean inOnesSection = false;
        boolean inZerosSection = false;
        int onesCount = 0;
        int zerosAfterOnes = 0;

        for (boolean value : binary) {
            if (value) {
                if (!inOnesSection) {
                    inOnesSection = true;
                }
                if (inZerosSection) {
                    // Found 1s after zeros - not a clean pulse
                    return false;
                }
                onesCount++;
            } else {
                if (inOnesSection) {
                    inZerosSection = true;
                }
                if (inZerosSection) {
                    zerosAfterOnes++;
                }
            }
        }

        // Valid pulse: ones followed by zeros (peak then decay)
        // Simple check — interval guard (500ms) handles duplicates
        return inOnesSection && inZerosSection &&
                onesCount >= 2 &&
                zerosAfterOnes >= 2;
    }

    private void onStepDetected(long timestamp) {
        // Enforce minimum step interval to reject duplicate detections
        if (timestamp - lastStepDetectionTime < MIN_STEP_INTERVAL_MS) {
            return;
        }

        // ---- Movement bout tracking ----
        // Reset bout if it's been too long since the last step
        if (movementBoutStartTime > 0 && (timestamp - lastStepDetectionTime) > BOUT_RESET_MS) {
            movementBoutStartTime = 0;
        }
        if (movementBoutStartTime == 0) {
            movementBoutStartTime = timestamp; // Start of a new movement bout
        }
        long boutDuration = timestamp - movementBoutStartTime;

        // Classify and compute step length
        float stepLength;
        if (boutDuration < BURST_THRESHOLD_MS) {
            // Short burst (<2s) — likely device movement, use small displacement
            stepLength = BURST_STEP_LENGTH;
            Log.d("StepDetection", String.format(
                    "Step [BURST] boutDuration=%dms → stepLength=%.2fm", boutDuration, stepLength));
        } else {
            // Sustained movement (>2s) — likely real walking
            stepLength = Math.min(calculateStepLength(), MAX_WALKING_STEP);
            Log.d("StepDetection", String.format(
                    "Step [WALK]  boutDuration=%dms → stepLength=%.2fm", boutDuration, stepLength));
        }

        // Increment counters (only if step is allowed)
        totalStepCount++;
        sessionStepCount++;

        // Update particle filter with prediction step
        if (particleFilter != null && particleFilter.isInitialized()) {
            // Calculate heading change since last step
            float headingChange = 0.0f;
            if (lastGyroTime > 0) {
                headingChange = currentHeading - previousStepHeading;

                // Normalize to [-π, π] to handle wraparound
                while (headingChange > Math.PI)
                    headingChange -= 2 * Math.PI;
                while (headingChange < -Math.PI)
                    headingChange += 2 * Math.PI;

                previousStepHeading = currentHeading;
            }

            particleFilter.predict(stepLength, headingChange);

            // Get updated position estimate
            fusedPosition = particleFilter.getEstimatedPosition();

            if (fusedPosition != null) {
                for (DeviceStateListener listener : listeners) {
                    listener.onPositionUpdated(fusedPosition);
                }

                Log.d("ParticleFilter", String.format(
                        "Position: (%.2f, %.2f) heading: %.1f° (Δ%.1f°) confidence: %.2f",
                        fusedPosition.x, fusedPosition.y,
                        Math.toDegrees(currentHeading), Math.toDegrees(headingChange),
                        fusedPosition.confidence));
            }
        }

        // Handle magnetic field initialization (first 10 steps)
        if (sessionStepCount <= INITIALIZATION_STEPS && !hasInitialPosition) {
            if (sessionStepCount == 1) {
                // Start collecting magnetic data for initialization
                magneticService.startInitialPositionEstimation();
                isInitializingPosition = true;
                Log.d("StepDetection", "Starting magnetic field initialization...");
            }

            // Notify magnetic service about step
            magneticService.onStepDetected();

            if (sessionStepCount == INITIALIZATION_STEPS) {
                // We've collected enough steps, process for initialization
                isInitializingPosition = false;
                Log.d("StepDetection", "Collected 10 steps for magnetic initialization");
            }
        }

        Log.d("StepDetection", String.format("Step #%d detected, length: %.2fm",
                totalStepCount, stepLength));

        // Notify listener
        if (listeners.isEmpty()) {
            Log.w("StepDetection", "⚠ No listeners registered - step update not sent to UI");
        } else {
            Log.d("StepDetection", String.format(
                    "Notifying %d listener(s) of step detection", listeners.size()));
        }
        for (DeviceStateListener listener : listeners) {
            listener.onStepDetected(timestamp, stepLength, totalStepCount, sessionStepCount);
        }

        // Notify calibration listener if in calibration mode
        if (calibrationMode && calibrationListener != null) {
            calibrationListener.onCalibrationStepDetected(sessionStepCount, binaryThreshold);
        }

        // Collect data for continuous learning
        collectStepData(timestamp, true); // true = actual step detected

        // Update particle filter with magnetic measurement (if available)
        if (particleFilter != null && magneticService != null && latestMag != null) {
            float[] magPos = magneticService.getCurrentPosition(latestMag);
            if (magPos != null && magPos.length >= 3) {
                com.example.wytv2.localization.Position magneticPosition = new com.example.wytv2.localization.Position(
                        magPos[0], magPos[1], (int) magPos[2], 0.7f);

                float confidence = 0.7f; // Moderate confidence for magnetic
                particleFilter.updateWithMagnetic(magneticPosition, confidence);

                Log.d("StepDetection", String.format(
                        "Magnetic correction: (%.2f, %.2f) floor %d, confidence: %.2f",
                        magPos[0], magPos[1], (int) magPos[2], confidence));
            }
        }

        // Check if retraining should be triggered
        if (retrainingScheduler != null) {
            retrainingScheduler.checkAndScheduleRetraining(totalStepCount);
        }
        // Reset for next step
        stepMaxAcc = 0;
        stepMinAcc = Float.MAX_VALUE;
        currentStepAccelerations.clear();
    }

    private float calculateStepLength() {
        if (currentStepAccelerations.size() < 5) {
            return 0.7f; // Default
        }

        // Calculate acceleration statistics
        float range = stepMaxAcc - stepMinAcc;
        if (range <= 0)
            return 0.7f;

        // Adaptive Weinberg model with frequency adjustment
        float baseLength = 0.7f;
        float adjustment = (float) Math.pow(range, 0.25);

        // Estimate frequency from step interval
        float frequency = 0;
        if (lastStepTime > 0 && lastStepDetectionTime > 0) {
            float interval = (lastStepDetectionTime - lastStepTime) / 1000.0f;
            if (interval > 0) {
                frequency = 1.0f / interval;
                // Adjust for frequency (higher frequency = shorter steps)
                adjustment *= (2.0f / (1 + frequency));
            }
        }

        float stepLength = baseLength * adjustment;

        // Apply reasonable bounds
        return Math.max(0.4f, Math.min(1.2f, stepLength));
    }

    private void resetStepTracking() {
        stepMaxAcc = 0;
        stepMinAcc = Float.MAX_VALUE;
        currentStepAccelerations.clear();
    }

    /**
     * Check if step candidate timestamps show consistent cadence (rhythmic
     * walking).
     * Uses coefficient of variation (stdDev / mean) of inter-step intervals.
     * Walking typically has CV < 15-20%; hand shaking is much higher.
     */
    private boolean isCadenceConsistent(List<Long> timestamps) {
        if (timestamps.size() < 2)
            return false;

        // Calculate inter-step intervals
        float[] intervals = new float[timestamps.size() - 1];
        float mean = 0;
        for (int i = 1; i < timestamps.size(); i++) {
            intervals[i - 1] = timestamps.get(i) - timestamps.get(i - 1);
            mean += intervals[i - 1];
        }
        mean /= intervals.length;

        // Reject if mean interval is outside reasonable walking range (300ms - 1200ms)
        if (mean < 300 || mean > 1200) {
            Log.d("StepDetection", String.format(
                    "Cadence rejected: mean interval %.0fms outside walking range [300-1200ms]", mean));
            return false;
        }

        // Calculate coefficient of variation
        float variance = 0;
        for (float interval : intervals) {
            variance += (interval - mean) * (interval - mean);
        }
        variance /= intervals.length;
        float stdDev = (float) Math.sqrt(variance);
        float cv = stdDev / mean;

        Log.d("StepDetection", String.format(
                "Cadence check: mean=%.0fms, stdDev=%.0fms, CV=%.2f (max=%.2f) → %s",
                mean, stdDev, cv, MAX_CADENCE_CV, cv < MAX_CADENCE_CV ? "PASS" : "FAIL"));

        return cv < MAX_CADENCE_CV;
    }

    private void updateStationaryDetection(float acceleration) {
        // Add to stationary buffer (absolute value)
        stationaryBuffer.add(Math.abs(acceleration));
        if (stationaryBuffer.size() > STATIONARY_BUFFER_SIZE) {
            stationaryBuffer.remove(0);
        }

        if (stationaryBuffer.size() >= 15) { // Need some data (was 20)
            float stdDev = calculateStdDev(stationaryBuffer);
            boolean newStationary = stdDev < STATIONARY_THRESHOLD;

            if (newStationary != isStationary) {
                boolean wasStationary = isStationary; // Store previous state for logging
                isStationary = newStationary;

                // Calculate confidence for state change
                float confidence = 1.0f - (stdDev / STATIONARY_THRESHOLD);

                // Drift prevention: Track state transitions
                if (isStationary && correctionEnabled) {
                    // Just became stationary
                    stationaryStartTime = System.currentTimeMillis();
                    driftPreventionResetPerformed = false;
                    if (particleFilter != null && particleFilter.isInitialized()) {
                        lastPositionBeforeStationary = particleFilter.getEstimatedPosition();
                        if (lastPositionBeforeStationary != null) {
                            Log.d("StepDetection", String.format(
                                    "Saved position before stationary: (%.2f, %.2f)",
                                    lastPositionBeforeStationary.x, lastPositionBeforeStationary.y));
                            runPositionCorrection(
                                    lastPositionBeforeStationary.x,
                                    lastPositionBeforeStationary.y);
                        }
                    }
                    driftPreventionHandler.postDelayed(this::checkAndPreventDrift, 5000);
                } else {
                    // Just started moving - clear saved position and stop drift prevention.
                    // NOTE: Do NOT zero stationaryStartTime here; onStartedMoving() reads
                    // it in the same call stack (processAccelerometer → updateStationaryDetection
                    // → onStartedMoving). Zeroing it here would make onStartedMoving() compute
                    // a billion-millisecond duration and falsely reset the session.
                    lastPositionBeforeStationary = null;
                    driftPreventionResetPerformed = false; // Reset flag for next stationary period
                    driftPreventionHandler.removeCallbacks(this::checkAndPreventDrift);
                }

                Log.d("StepDetection", String.format(
                        "State: %s → %s (confidence: %.2f)",
                        wasStationary ? "STATIONARY" : "MOVING",
                        isStationary ? "STATIONARY" : "MOVING", confidence));

                if (listeners.isEmpty()) {
                    Log.w("StepDetection", "⚠ No listeners registered - state change not sent to UI");
                } else {
                    Log.d("StepDetection", String.format(
                            "Notifying %d listener(s) of state change", listeners.size()));
                }
                for (DeviceStateListener listener : listeners) {
                    listener.onDeviceStateChanged(isStationary, confidence);
                }

                Log.d("StepDetection", "State: " +
                        (isStationary ? "STATIONARY" : "MOVING") +
                        " (stdDev: " + String.format("%.3f", stdDev) + ")");
            }
        }
    }

    /**
     * Check and prevent position drift during stationary state.
     * 
     * TEMPORARILY DISABLED: Re-initialization destroys position history.
     * The particle filter was being reset every 20-30 seconds during brief pauses,
     * which destroyed the accumulated position estimate and caused severe accuracy
     * issues.
     * 
     * TODO: Implement position constraint instead of full re-initialization.
     * A proper solution would nudge particles back without destroying filter state.
     */
    private void checkAndPreventDrift() {
        // DISABLED: Re-initialization causes more problems than it solves
        // Root cause: Brief pauses during walking trigger resets, destroying position
        // history
        // Evidence: Logs show "Initialized with 500 particles" every 20-30 seconds
        return;

        /*
         * ORIGINAL CODE (DISABLED):
         * if (!isStationary || particleFilter == null ||
         * !particleFilter.isInitialized()) {
         * return;
         * }
         * 
         * long stationaryDuration = System.currentTimeMillis() - stationaryStartTime;
         * 
         * // Only reset if we haven't already done so in this stationary period
         * if (stationaryDuration >= STATIONARY_RESET_DELAY_MS &&
         * lastPositionBeforeStationary != null &&
         * !driftPreventionResetPerformed) {
         * 
         * // Reset particle filter to last known position before stationary
         * particleFilter.initialize(lastPositionBeforeStationary, 500, currentHeading);
         * 
         * Log.d("StepDetection", String.format(
         * "Drift prevention: Reset position to (%.2f, %.2f) after %.1fs stationary",
         * lastPositionBeforeStationary.x, lastPositionBeforeStationary.y,
         * stationaryDuration / 1000.0f));
         * 
         * // Update fused position
         * fusedPosition = particleFilter.getEstimatedPosition();
         * if (fusedPosition != null && stateListener != null) {
         * stateListener.onPositionUpdated(fusedPosition);
         * }
         * 
         * // Mark that we've performed the reset
         * driftPreventionResetPerformed = true;
         * }
         * 
         * // Schedule next check (every 5 seconds while stationary)
         * // This continues to run but won't reset again due to the flag
         * if (isStationary) {
         * driftPreventionHandler.postDelayed(this::checkAndPreventDrift, 5000);
         * }
         */
    }
    // =========================================================================
    // Position correction (P0→P3) — shared by stationary detection + manual
    // calibration
    // =========================================================================

    /**
     * Run the priority-ordered position-correction chain (P0→P3) from the given
     * PDR coordinate, then notify all registered listeners with the result.
     */
    private void runPositionCorrection(float curX, float curY) {
        if (particleFilter == null || !particleFilter.isInitialized())
            return;

        // Snapshot WiFi + magnetic readings
        List<com.example.wytv2.wifi.WiFiReading> wifiReadings = (wifiService != null) ? wifiService.getCurrentReadings()
                : null;
        Map<String, Integer> curWifi = new java.util.HashMap<>();
        if (wifiReadings != null) {
            for (com.example.wytv2.wifi.WiFiReading r : wifiReadings)
                curWifi.put(r.bssid, r.rssi);
        }
        float curMag = (float) Math.sqrt(
                latestMag[0] * latestMag[0] +
                        latestMag[1] * latestMag[1] +
                        latestMag[2] * latestMag[2]);

        boolean corrected = false;
        final float ZONE_ANCHOR_PROMOTE_M = 3.0f;
        final float P0_MIN_WIFI_SIM = 0.85f;

        // ---- P0: Zone-fingerprint snap ----
        if (!zonePositions.isEmpty() && !sensorAnchors.isEmpty() && !curWifi.isEmpty()) {
            SensorAnchor p0Best = null;
            float p0BestSim = 0;
            int p0ZoneIdx = -1;
            for (SensorAnchor anchor : sensorAnchors) {
                float magDiff = Math.abs(curMag - anchor.magMagnitude);
                if (magDiff > ANCHOR_MAG_DIFF_THRESHOLD)
                    continue;
                float dot = 0, normA = 0, normB = 0;
                for (Map.Entry<String, Integer> e : curWifi.entrySet()) {
                    int rssiA = e.getValue();
                    Integer rssiB = anchor.wifiRssi.get(e.getKey());
                    if (rssiB != null) {
                        dot += rssiA * rssiB;
                        normA += rssiA * rssiA;
                        normB += rssiB * rssiB;
                    }
                }
                if (normA <= 0 || normB <= 0)
                    continue;
                float sim = dot / (float) (Math.sqrt(normA) * Math.sqrt(normB));
                if (sim < P0_MIN_WIFI_SIM)
                    continue;
                float ax = anchor.pdrPosition[0], ay = anchor.pdrPosition[1];
                for (int zi = 0; zi < zonePositions.size(); zi++) {
                    float[] zp2 = zonePositions.get(zi);
                    float d2z = (float) Math.sqrt(Math.pow(ax - zp2[0], 2) + Math.pow(ay - zp2[1], 2));
                    if (d2z <= ZONE_ANCHOR_PROMOTE_M && sim > p0BestSim) {
                        p0BestSim = sim;
                        p0Best = anchor;
                        p0ZoneIdx = zi;
                    }
                }
            }
            if (p0Best != null && p0ZoneIdx >= 0) {
                int cz = nearestZoneIndex(curX, curY);
                int az = nearestZoneIndex(p0Best.pdrPosition[0], p0Best.pdrPosition[1]);
                if (zonePositions.size() == 1 || cz == az) {
                    float[] sz = zonePositions.get(p0ZoneIdx);
                    Log.d("StepDetection",
                            String.format("P0 SNAP: sim=%.2f zone%d (%.2f,%.2f)", p0BestSim, p0ZoneIdx, sz[0], sz[1]));
                    particleFilter.initialize(new Position(sz[0], sz[1], 0), 500, currentHeading, 0.3f);
                    corrected = true;
                }
            }
        }

        // ---- P1: Zone-center snap (within 5m) ----
        if (!corrected && !zonePositions.isEmpty()) {
            int nzi = nearestZoneIndex(curX, curY);
            float[] zp = zonePositions.get(nzi);
            float dist = (float) Math.sqrt(Math.pow(curX - zp[0], 2) + Math.pow(curY - zp[1], 2));
            if (dist <= 5.0f) {
                Log.d("StepDetection", String.format("P1 SNAP: %.1fm zone%d (%.2f,%.2f)", dist, nzi, zp[0], zp[1]));
                particleFilter.initialize(new Position(zp[0], zp[1], 0), 500, currentHeading, 0.3f);
                corrected = true;
            }
        }

        // ---- P2: Dynamic anchor fingerprint snap ----
        if (!corrected) {
            SensorAnchor bestMatch = null;
            float bestSim = 0;
            for (SensorAnchor anchor : sensorAnchors) {
                float magDiff = Math.abs(curMag - anchor.magMagnitude);
                if (magDiff > ANCHOR_MAG_DIFF_THRESHOLD)
                    continue;
                if (!curWifi.isEmpty() && !anchor.wifiRssi.isEmpty()) {
                    float dot = 0, normA = 0, normB = 0;
                    for (Map.Entry<String, Integer> e : curWifi.entrySet()) {
                        int rssiA = e.getValue();
                        Integer rssiB = anchor.wifiRssi.get(e.getKey());
                        if (rssiB != null) {
                            dot += rssiA * rssiB;
                            normA += rssiA * rssiA;
                            normB += rssiB * rssiB;
                        }
                    }
                    if (normA > 0 && normB > 0) {
                        float sim = dot / (float) (Math.sqrt(normA) * Math.sqrt(normB));
                        if (sim > ANCHOR_WIFI_SIM_THRESHOLD && sim > bestSim) {
                            bestSim = sim;
                            bestMatch = anchor;
                        }
                    }
                } else if (curWifi.isEmpty()) {
                    float sim = 1.0f - (Math.abs(curMag - anchor.magMagnitude) / ANCHOR_MAG_DIFF_THRESHOLD);
                    if (sim > bestSim) {
                        bestSim = sim;
                        bestMatch = anchor;
                    }
                }
            }
            if (bestMatch != null) {
                int cz = nearestZoneIndex(curX, curY);
                int az = nearestZoneIndex(bestMatch.pdrPosition[0], bestMatch.pdrPosition[1]);
                if (zonePositions.isEmpty() || cz == az) {
                    float ax2 = bestMatch.pdrPosition[0], ay2 = bestMatch.pdrPosition[1];
                    Position snapPos = new Position(ax2, ay2, 0);
                    for (float[] zp3 : zonePositions) {
                        if ((float) Math
                                .sqrt(Math.pow(ax2 - zp3[0], 2) + Math.pow(ay2 - zp3[1], 2)) <= ZONE_ANCHOR_PROMOTE_M) {
                            snapPos = new Position(zp3[0], zp3[1], 0);
                            break;
                        }
                    }
                    Log.d("StepDetection",
                            String.format("P2 SNAP: sim=%.2f → (%.2f,%.2f)", bestSim, snapPos.x, snapPos.y));
                    particleFilter.initialize(snapPos, 500, currentHeading, 0.3f);
                    corrected = true;
                }
            }
            // Save dynamic anchor in unzoned space
            if (!corrected || bestMatch == null) {
                boolean inZone = false;
                for (float[] zp4 : zonePositions) {
                    if (Math.sqrt(Math.pow(curX - zp4[0], 2) + Math.pow(curY - zp4[1], 2)) <= 5.0f) {
                        inZone = true;
                        break;
                    }
                }
                if (!inZone) {
                    boolean tooClose = false;
                    for (SensorAnchor anc : sensorAnchors) {
                        float dx = curX - anc.pdrPosition[0], dy = curY - anc.pdrPosition[1];
                        if (Math.sqrt(dx * dx + dy * dy) < ANCHOR_PDR_MIN_SEPARATION) {
                            tooClose = true;
                            break;
                        }
                    }
                    if (!tooClose) {
                        sensorAnchors.add(new SensorAnchor(curWifi, curMag, curX, curY));
                        if (sensorAnchors.size() > MAX_ANCHORS)
                            sensorAnchors.remove(0);
                        Log.d("StepDetection",
                                String.format("Anchor saved (%.2f,%.2f) [%d]", curX, curY, sensorAnchors.size()));
                    }
                }
            }
            // ---- P3: Nearest-zone fallback ----
            if (!corrected && !zonePositions.isEmpty()) {
                int ni = nearestZoneIndex(curX, curY);
                float[] zp = zonePositions.get(ni);
                Log.d("StepDetection", String.format("P3 SNAP: zone%d (%.2f,%.2f)", ni, zp[0], zp[1]));
                particleFilter.initialize(new Position(zp[0], zp[1], 0), 500, currentHeading, 0.5f);
                corrected = true;
            }
        }

        // Notify listeners
        fusedPosition = particleFilter.getEstimatedPosition();
        if (fusedPosition != null) {
            for (DeviceStateListener l : listeners)
                l.onPositionUpdated(fusedPosition);
            Log.d("StepDetection", String.format("Correction done: (%.2f,%.2f) corrected=%b", fusedPosition.x,
                    fusedPosition.y, corrected));
        }
    }

    /**
     * Manually trigger the P0→P3 position-correction chain on demand.
     * Safe to call from any thread. Uses current particle-filter estimate as
     * origin.
     * Intended for use by the Calibrate Floor button in MainActivity.
     */
    public void triggerStationaryCorrection() {
        if (particleFilter == null || !particleFilter.isInitialized()) {
            Log.w("StepDetection", "triggerStationaryCorrection: filter not ready");
            return;
        }
        com.example.wytv2.localization.Position cur = particleFilter.getEstimatedPosition();
        if (cur == null) {
            Log.w("StepDetection", "triggerStationaryCorrection: no position");
            return;
        }
        Log.d("StepDetection", String.format("Manual snap from (%.2f,%.2f)", cur.x, cur.y));
        runPositionCorrection(cur.x, cur.y);
    }

    private void onStartedMoving() {
        // Only reset session steps if we have a valid stationaryStartTime (> 0) AND
        // the device was actually stationary for more than 5 minutes.
        // Guard prevents a race: stationaryStartTime=0 at field init would produce a
        // billions-of-ms duration on the very first movement, falsely resetting the
        // session.
        if (stationaryStartTime > 0) {
            long stationaryDuration = System.currentTimeMillis() - stationaryStartTime;
            if (stationaryDuration > 300000) { // 5 minutes in milliseconds
                Log.d("StepDetection", String.format(
                        "Started moving after long stationary period (%.1f min) - resetting session",
                        stationaryDuration / 60000.0));
                sessionStepCount = 0;
            } else {
                Log.d("StepDetection", String.format(
                        "Started moving after brief pause (%.1f sec) - keeping session steps (%d)",
                        stationaryDuration / 1000.0, sessionStepCount));
            }
        }
        stationaryStartTime = 0; // Reset here, after reading, not before

        hasInitialPosition = false;
        initialPosition = null;
        matchedCalibrationNodes.clear();

        // Clear buffers for fresh start
        rawAccelerationBuffer.clear();
        smoothedAccelerationBuffer.clear();
        stationaryBuffer.clear();
        stepCandidateTimestamps.clear();
        stepsValidated = false;
        resetStepTracking();
        lastStepTime = 0;
        lastStepDetectionTime = System.currentTimeMillis();

        for (DeviceStateListener listener : listeners) {
            listener.onStepsReset();
        }
    }

    private float calculateStdDev(List<Float> buffer) {
        if (buffer.isEmpty())
            return 0;

        float mean = 0;
        for (float val : buffer)
            mean += val;
        mean /= buffer.size();

        float sumSq = 0;
        for (float val : buffer) {
            sumSq += (val - mean) * (val - mean);
        }

        return (float) Math.sqrt(sumSq / buffer.size());
    }

    // Public methods
    public int getTotalStepCount() {
        return totalStepCount;
    }

    public int getSessionStepCount() {
        return sessionStepCount;
    }

    public boolean isStationary() {
        return isStationary;
    }

    public float getBinaryThreshold() {
        return binaryThreshold;
    }

    public boolean hasInitialPosition() {
        return hasInitialPosition;
    }

    public float[] getInitialPosition() {
        return initialPosition;
    }

    public List<CalibrationNode> getMatchedCalibrationNodes() {
        return matchedCalibrationNodes;
    }

    public void setBinaryThreshold(float threshold) {
        this.binaryThreshold = threshold;
        Log.d("StepDetection", "Threshold set to: " + threshold);
    }

    public int getWindowSize() {
        return windowSize;
    }

    public void setWindowSize(int size) {
        this.windowSize = size;
    }

    public void resetAllSteps() {
        totalStepCount = 0;
        sessionStepCount = 0;
        hasInitialPosition = false;
        initialPosition = null;
        matchedCalibrationNodes.clear();

        for (DeviceStateListener listener : listeners) {
            listener.onStepsReset();
        }
        Log.d("StepDetection", "All steps reset");
    }

    // Debug methods
    public float getLatestAcceleration() {
        return latestAcceleration;
    }

    public int getBufferSize() {
        return rawAccelerationBuffer.size();
    }

    public long getTimeSinceLastStep() {
        return System.currentTimeMillis() - lastStepDetectionTime;
    }

    // Magnetic service access
    public MagneticFieldLocalizationService getMagneticService() {
        return magneticService;
    }

    public void loadMagneticDatabase(String buildingName) {
        magneticService.loadFingerprintDatabase(buildingName);
    }

    // ML Model Integration Methods

    /**
     * Add current sensor readings to feature buffer and run prediction if ready.
     */
    private void addToFeatureBuffer(long timestamp) {
        // Add sensor reading to buffer
        featureBuffer.addReading(latestAccel, latestGyro, latestMag, timestamp);

        // Feed sensor data to ML WiFi positioning model
        if (wifiPositioningModel != null && wifiPositioningModel.isModelLoaded()) {
            wifiPositioningModel.addSensorReading(latestAccel, latestGyro, latestMag);

            // Try to get a position prediction
            com.example.wytv2.ml.WiFiPositioningModel.PositionPrediction mlPos = wifiPositioningModel.predict();
            if (mlPos != null) {
                if (mlPos.hasDelta && particleFilter != null && particleFilter.isInitialized()) {
                    // Apply relative displacement as drift correction
                    // Scale: UJIIndoorLoc units → approx meters (empirical factor)
                    // Alpha: 0.3 = apply 30% of the ML-suggested correction
                    float ML_COORD_SCALE = 0.001f; // UJIIndoorLoc units are ~1000x meters
                    float ML_CORRECTION_ALPHA = 0.3f;

                    particleFilter.applyMLDriftCorrection(
                            mlPos.deltaX, mlPos.deltaY,
                            ML_COORD_SCALE, ML_CORRECTION_ALPHA);

                    // Get updated position after correction
                    fusedPosition = particleFilter.getEstimatedPosition();
                    if (fusedPosition != null) {
                        for (DeviceStateListener listener : listeners) {
                            listener.onPositionUpdated(fusedPosition);
                        }
                    }

                    Log.i("WiFiML", String.format(
                            "ML drift correction applied: Δ=(%.4f, %.4f) → scaled correction",
                            mlPos.deltaX, mlPos.deltaY));
                } else {
                    Log.d("WiFiML", String.format(
                            "ML baseline prediction: (%.2f, %.2f) floor %d",
                            mlPos.x, mlPos.y, mlPos.floor));
                }

                // Forward ML prediction to listeners
                for (DeviceStateListener listener : listeners) {
                    listener.onMLPositionPrediction(mlPos.x, mlPos.y, mlPos.floor);
                }
            }
        }

        // Check if it's time to run activity prediction
        if (timestamp - lastActivityPredictionTime >= ACTIVITY_PREDICTION_INTERVAL_MS) {
            runActivityPrediction(timestamp);
        }
    }

    /**
     * Run ML model inference on buffered sensor data.
     */
    private void runActivityPrediction(long timestamp) {
        if (!featureBuffer.isReady()) {
            return; // Not enough data yet
        }

        // Extract features (full 50x69 window)
        float[][] features = featureBuffer.extractFeatures();
        if (features == null || features.length == 0) {
            return;
        }

        // Predict activity using TFLite model (or rule-based fallback)
        SimplifiedActivityModel.PredictionResult result = SimplifiedActivityModel.predictWithConfidence(features);

        currentActivity = result.activity;
        currentActivityConfidence = result.confidence;
        lastActivityPredictionTime = timestamp;

        // Adaptive threshold adjustment based on activity
        adjustThresholdForActivity(currentActivity, currentActivityConfidence);

        // Log prediction with all class probabilities
        Log.d("StepDetection", String.format(
                "Activity: %s (%.1f%%) | Stat=%.2f Walk=%.2f Run=%.2f Up=%.2f Down=%.2f - Threshold: %.2f [%s]",
                currentActivity, currentActivityConfidence * 100,
                result.probabilities[0], result.probabilities[1], result.probabilities[2],
                result.probabilities[3], result.probabilities[4],
                binaryThreshold,
                SimplifiedActivityModel.isOnnxModelLoaded() ? "ONNX" : "Rules"));

        // Notify listener
        for (DeviceStateListener listener : listeners) {
            listener.onActivityRecognized(currentActivity, currentActivityConfidence);
        }
    }

    /**
     * Adjust step detection threshold based on predicted activity.
     * Higher thresholds for non-walking activities to reduce false positives.
     */
    private void adjustThresholdForActivity(String activity, float confidence) {
        // Only adjust if confidence is reasonable
        if (confidence < 0.5f) {
            return; // Not confident enough
        }

        float newThreshold = binaryThreshold;

        switch (activity) {
            case "Walking":
            case "Running":
            case "Stairs Up":
            case "Stairs Down":
                // Low threshold for movement activities - maximize sensitivity
                newThreshold = 0.9f;
                break;

            case "Stationary":
                // High threshold for stationary - reduce false positives
                newThreshold = 1.8f;
                break;

            default:
                // Unknown activity - use moderate threshold
                newThreshold = 1.3f;
                break;
        }

        // Smooth threshold changes
        binaryThreshold = binaryThreshold * 0.7f + newThreshold * 0.3f;
    }

    /**
     * Determine if step detection should be allowed based on current activity.
     * This is the KEY method for reducing false positives from hand gestures!
     *
     * @param activity   Current detected activity
     * @param confidence Confidence of the prediction
     * @return true if steps should be counted, false if they should be blocked
     */
    private boolean isStepAllowedForActivity(String activity, float confidence) {
        // Require minimum confidence for any decision
        if (confidence < 0.5f) {
            return true; // Low confidence - allow by default to avoid blocking real steps
        }

        // ALLOW STEPS for movement activities
        if ((activity.equals("Walking") || activity.equals("Running") ||
                activity.equals("Stairs Up") || activity.equals("Stairs Down"))
                && confidence > 0.5f) {
            return true;
        }

        // BLOCK STEPS for stationary activity
        if (activity.equals("Stationary") && confidence > 0.7f) {
            return false; // High confidence stationary - block steps
        }

        // Default: allow steps (err on the side of not blocking real walking)
        return true;
    }

    /**
     * Get the current recognized activity.
     */
    public String getCurrentActivity() {
        return currentActivity;
    }

    /**
     * Get the confidence of the current activity prediction.
     */
    public float getCurrentActivityConfidence() {
        return currentActivityConfidence;
    }

    private void checkFalsePositiveWithWiFi() {
        if (wifiRssiVariance < RSSI_STABLE_THRESHOLD) {
            float wifiAdjustedThreshold = Math.max(binaryThreshold, 1.7f);
            if (Math.abs(wifiAdjustedThreshold - binaryThreshold) > 0.05f) {
                binaryThreshold = wifiAdjustedThreshold;
                Log.d("StepDetection", String.format(
                        "WiFi stable (variance: %.2f) - increased threshold to %.2f",
                        wifiRssiVariance, binaryThreshold));
            }
        }
    }

    public com.example.wytv2.wifi.WiFiFingerprintDatabase getWiFiDatabase() {
        return wifiDatabase;
    }

    public float getWiFiRSSIVariance() {
        return wifiRssiVariance;
    }

    /**
     * Collect step data for continuous learning.
     */
    private void collectStepData(long timestamp, boolean wasActualStep) {
        if (dataCollector == null || !dataCollector.isCollectingEnabled()) {
            return;
        }

        // Extract features from buffer
        float[][] features = featureBuffer.extractFeatures();
        if (features == null || features.length == 0) {
            return;
        }

        // Get most recent feature vector
        float[] featureVector = features[features.length - 1];

        // Determine source
        String source = calibrationMode ? "calibration" : "normal";

        // Record data point
        dataCollector.recordStepCandidate(
                latestAccel.clone(),
                latestGyro.clone(),
                latestMag.clone(),
                featureVector,
                wasActualStep,
                binaryThreshold,
                currentActivity,
                currentActivityConfidence,
                source);
    }

    /**
     * Get retraining scheduler for UI access.
     */
    public com.example.wytv2.ml.RetrainingScheduler getRetrainingScheduler() {
        return retrainingScheduler;
    }

    /**
     * Get model retrainer for UI access.
     */
    public com.example.wytv2.ml.ModelRetrainer getModelRetrainer() {
        if (retrainingScheduler == null)
            return null;
        return new com.example.wytv2.ml.ModelRetrainer(context, dataCollector);
    }

    /**
     * Get data collector for external access.
     */
    public com.example.wytv2.ml.StepDataCollector getDataCollector() {
        return dataCollector;
    }

    /**
     * Get current heading from gyroscope integration.
     */
    public float getCurrentHeading() {
        return currentHeading;
    }

    /**
     * Get particle filter for external access (e.g., testing UI).
     */
    public com.example.wytv2.localization.ParticleFilterLocalization getParticleFilter() {
        return particleFilter;
    }

}
