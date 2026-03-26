package com.example.wytv2;

import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.wytv2.pdr.StepDetectionService;

public class SettingsActivity extends AppCompatActivity {

    private StepDetectionService stepDetectionService;
    private StepDetectionService.DeviceStateListener settingsListener;

    // UI Elements
    private TextView tvStepCount;
    private TextView tvState;
    private TextView tvThresholdValue;
    private Button btnResetSteps;
    private Button btnDecrease;
    private Button btnIncrease;

    // ML Debug Panel UI
    private TextView tvMlBufferStatus;
    private TextView tvMlActivityPrediction;
    private TextView tvMlThresholdStatus;
    private TextView tvMlLastUpdate;

    // WiFi RSSI Panel UI
    private TextView tvWifiApCount;
    private TextView tvWifiVariance;
    private TextView tvWifiStatus;
    private TextView tvWifiStrongestAp;

    // Continuous Learning UI
    private TextView tvRetrainingProgress;
    private android.widget.ProgressBar retrainingProgressBar;
    private TextView tvRetrainingStats;
    private TextView tvRetrainingLastRun;

    // Particle Filter Position UI
    private TextView tvPositionStatus;
    private TextView tvPositionCoordinates;
    private TextView tvPositionHeading;
    private TextView tvPositionConfidence;

    // Particle Filter Testing UI
    private Button btnStartPfTest;
    private TextView tvPfTestStatus;
    private TextView tvPfMeanError;
    private TextView tvPfDebug;
    private com.example.wytv2.localization.ParticleFilterTester particleFilterTester;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);

        // Get step detection service from MainActivity
        stepDetectionService = MainActivity.getStepDetectionService();

        if (stepDetectionService == null) {
            Log.e("SettingsActivity", "✗ StepDetectionService is NULL - UI will not update!");
            Log.e("SettingsActivity", "  This usually means MainActivity hasn't initialized the service yet.");
        } else {
            Log.d("SettingsActivity", "✓ StepDetectionService retrieved successfully");
        }

        // Initialize UI
        initUI();

        // Set up listeners
        setupListeners();

        // Initialize service listener (create the listener object)
        initServiceListener();

        // DON'T register listener here - let onResume() handle it for clean lifecycle
        // management
    }

    private void initUI() {
        // Basic UI
        tvStepCount = findViewById(R.id.step_count);
        tvState = findViewById(R.id.device_state);
        tvThresholdValue = findViewById(R.id.threshold_value);
        btnDecrease = findViewById(R.id.decrease_threshold);
        btnResetSteps = findViewById(R.id.reset_steps);
        btnIncrease = findViewById(R.id.increase_threshold);

        // ML Debug Panel
        tvMlBufferStatus = findViewById(R.id.ml_buffer_status);
        tvMlActivityPrediction = findViewById(R.id.ml_activity_prediction);
        tvMlThresholdStatus = findViewById(R.id.ml_threshold_status);
        tvMlLastUpdate = findViewById(R.id.ml_last_update);

        // WiFi RSSI Panel
        tvWifiApCount = findViewById(R.id.wifi_ap_count);
        tvWifiVariance = findViewById(R.id.wifi_variance);
        tvWifiStatus = findViewById(R.id.wifi_status);
        tvWifiStrongestAp = findViewById(R.id.wifi_strongest_ap);

        // Continuous Learning UI
        tvRetrainingProgress = findViewById(R.id.retraining_progress);
        retrainingProgressBar = findViewById(R.id.retraining_progress_bar);
        tvRetrainingStats = findViewById(R.id.retraining_stats);
        tvRetrainingLastRun = findViewById(R.id.retraining_last_run);

        // Particle Filter Position UI
        tvPositionStatus = findViewById(R.id.position_status);
        tvPositionCoordinates = findViewById(R.id.position_coordinates);
        tvPositionHeading = findViewById(R.id.position_heading);
        tvPositionConfidence = findViewById(R.id.position_confidence);

        // Particle Filter Testing UI
        btnStartPfTest = findViewById(R.id.btn_start_pf_test);
        tvPfTestStatus = findViewById(R.id.tv_pf_test_status);
        tvPfMeanError = findViewById(R.id.tv_pf_mean_error);
        tvPfDebug = findViewById(R.id.tv_pf_debug);

        // Initialize tester
        particleFilterTester = new com.example.wytv2.localization.ParticleFilterTester();
    }

    private void setupListeners() {
        // Threshold controls
        if (btnDecrease != null) {
            btnDecrease.setOnClickListener(v -> {
                if (stepDetectionService != null) {
                    float currentThreshold = stepDetectionService.getBinaryThreshold();
                    float newThreshold = currentThreshold - 0.1f;
                    if (newThreshold >= 0.5f) {
                        stepDetectionService.setBinaryThreshold(newThreshold);
                        updateThresholdDisplay(newThreshold);
                    }
                }
            });
        }

        if (btnIncrease != null) {
            btnIncrease.setOnClickListener(v -> {
                if (stepDetectionService != null) {
                    float currentThreshold = stepDetectionService.getBinaryThreshold();
                    float newThreshold = currentThreshold + 0.1f;
                    if (newThreshold <= 2.5f) {
                        stepDetectionService.setBinaryThreshold(newThreshold);
                        updateThresholdDisplay(newThreshold);
                    }
                }
            });
        }

        // Reset steps button
        if (btnResetSteps != null) {
            btnResetSteps.setOnClickListener(v -> {
                if (stepDetectionService != null) {
                    stepDetectionService.resetAllSteps();
                    updateStepDisplay(0, 0);
                }
            });
        }

        // Calibration button
        Button calibrateButton = findViewById(R.id.calibrate_button);
        if (calibrateButton != null) {
            calibrateButton.setOnClickListener(v -> {
                android.content.Intent intent = new android.content.Intent(this, CalibrationActivity.class);
                startActivity(intent);
            });
        }

        // Particle filter test button
        if (btnStartPfTest != null) {
            btnStartPfTest.setOnClickListener(v -> {
                if (!particleFilterTester.isTracking()) {
                    // Start test
                    if (stepDetectionService != null) {
                        com.example.wytv2.localization.ParticleFilterLocalization pf = stepDetectionService
                                .getParticleFilter();
                        if (pf != null) {
                            com.example.wytv2.localization.Position startPos = new com.example.wytv2.localization.Position(
                                    0.0f, 0.0f, 1, 1.0f);
                            pf.initialize(startPos, 500);
                            Log.d("SettingsActivity", "✓ Particle filter reset to (0,0) for accuracy test");
                        }
                    }

                    particleFilterTester.startTracking();
                    btnStartPfTest.setText("Stop Test");
                    btnStartPfTest.setBackgroundTintList(
                            android.content.res.ColorStateList.valueOf(0xFFFF5722));
                    if (tvPfTestStatus != null) {
                        tvPfTestStatus.setText("Status: Testing (Walk 10 steps forward)");
                    }
                } else {
                    // Stop test
                    particleFilterTester.stopTracking();
                    btnStartPfTest.setText("Start Accuracy Test");
                    btnStartPfTest.setBackgroundTintList(
                            android.content.res.ColorStateList.valueOf(0xFF2196F3));
                    if (tvPfTestStatus != null) {
                        tvPfTestStatus.setText("Status: Test Complete");
                    }
                }
            });
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.d("SettingsActivity", "onResume() called");

        // Re-fetch service in case it was recreated
        stepDetectionService = MainActivity.getStepDetectionService();

        // Update UI with current values
        updateUI();

        if (stepDetectionService != null && settingsListener != null) {
            stepDetectionService.addDeviceStateListener(settingsListener);
            Log.d("SettingsActivity", "✓ Listener re-registered in onResume()");

            // Fetch current values to update UI
            updateUIWithCurrentValues();
        } else {
            Log.e("SettingsActivity", "✗ Cannot re-register listener in onResume()");
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        Log.d("SettingsActivity", "onStart() called");

        // Ensure service reference is current
        stepDetectionService = MainActivity.getStepDetectionService();

        // DON'T register listener here - let onResume() handle it for clean lifecycle
        // management
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.d("SettingsActivity", "onPause() called");
        if (stepDetectionService != null && settingsListener != null) {
            stepDetectionService.removeDeviceStateListener(settingsListener);
            Log.d("SettingsActivity", "✓ Listener unregistered in onPause()");
        }
    }

    /**
     * Fetch current values from service and update UI immediately.
     * This ensures UI shows current state when activity becomes visible.
     */
    private void updateUIWithCurrentValues() {
        if (stepDetectionService == null) {
            Log.e("SettingsActivity", "Cannot update UI - service is null");
            return;
        }

        try {
            // Get current step counts
            int totalSteps = stepDetectionService.getTotalStepCount();
            int sessionSteps = stepDetectionService.getSessionStepCount();

            Log.d("SettingsActivity", String.format(
                    "Fetched from service: total=%d, session=%d", totalSteps, sessionSteps));

            updateStepDisplay(totalSteps, sessionSteps);

            // Get current device state
            boolean isStationary = stepDetectionService.isStationary();
            String stateText = isStationary ? "Stationary (Stable)" : "Moving";

            Log.d("SettingsActivity", "Fetched device state: " + stateText);

            updateDeviceState(stateText);

            // Get current threshold
            float threshold = stepDetectionService.getBinaryThreshold();
            updateThresholdDisplay(threshold);

            // Update retraining progress
            updateRetrainingProgress();

            Log.d("SettingsActivity", "✓ UI updated with current values");

        } catch (Exception e) {
            Log.e("SettingsActivity", "Error updating UI with current values", e);
        }
    }

    private void updateUI() {
        if (stepDetectionService == null) {
            return;
        }

        // Update threshold display
        float threshold = stepDetectionService.getBinaryThreshold();
        updateThresholdDisplay(threshold);

        // Update retraining progress
        updateRetrainingProgress();
    }

    private void updateThresholdDisplay(float threshold) {
        if (tvThresholdValue != null) {
            tvThresholdValue.setText(String.format("Threshold: %.2f m/s²", threshold));
        }
        if (tvMlThresholdStatus != null) {
            tvMlThresholdStatus.setText(String.format("⚙️ Threshold: %.2f m/s²", threshold));
        }
    }

    private void updateStepDisplay(int totalSteps, int sessionSteps) {
        if (tvStepCount != null) {
            tvStepCount.setText(String.format("Steps\nTotal: %d\nSession: %d",
                    totalSteps, sessionSteps));
        }
    }

    private void updateRetrainingProgress() {
        if (stepDetectionService == null)
            return;

        try {
            com.example.wytv2.ml.RetrainingScheduler scheduler = stepDetectionService.getRetrainingScheduler();
            if (scheduler == null)
                return;

            com.example.wytv2.ml.RetrainingScheduler.RetrainingStats stats = scheduler.getStats();

            int interval = scheduler.getRetrainingInterval();
            int stepsRemaining = interval - stats.stepsSinceLastTraining;
            if (stepsRemaining < 0)
                stepsRemaining = 0;

            if (tvRetrainingProgress != null) {
                tvRetrainingProgress.setText(String.format(
                        "Next retraining in: %d steps", stepsRemaining));
            }

            if (retrainingProgressBar != null) {
                retrainingProgressBar.setMax(interval);
                retrainingProgressBar.setProgress(stats.stepsSinceLastTraining);
            }

            com.example.wytv2.ml.StepDataCollector dataCollector = stepDetectionService.getDataCollector();

            if (dataCollector != null && tvRetrainingStats != null) {
                com.example.wytv2.ml.StepDataCollector.DataStatistics dataStats = dataCollector.getStatistics();

                com.example.wytv2.ml.ModelRetrainer retrainer = stepDetectionService.getModelRetrainer();

                String accuracyStr = "--";
                if (retrainer != null) {
                    float accuracy = retrainer.getCurrentAccuracy();
                    accuracyStr = String.format("%.1f%%", accuracy * 100);
                }

                tvRetrainingStats.setText(String.format(
                        "Collected: %d samples | Accuracy: %s",
                        dataStats.totalSamples, accuracyStr));
            }

            if (tvRetrainingLastRun != null) {
                tvRetrainingLastRun.setText(String.format(
                        "Last retraining: %s", stats.getLastTrainingTimeString()));
            }

        } catch (Exception e) {
            Log.e("SettingsActivity", "Error updating retraining progress", e);
        }
    }

    // Public methods for MainActivity to update UI
    public void updateDeviceState(String state) {
        runOnUiThread(() -> {
            if (tvState != null) {
                tvState.setText(state);
            }
        });
    }

    public void updateWiFiPanel(java.util.List<com.example.wytv2.wifi.WiFiReading> readings, float variance) {
        runOnUiThread(() -> {
            if (readings == null || readings.isEmpty()) {
                if (tvWifiApCount != null) {
                    tvWifiApCount.setText("Access Points: 0");
                }
                if (tvWifiStatus != null) {
                    tvWifiStatus.setText("Status: No WiFi detected");
                }
                return;
            }

            if (tvWifiApCount != null) {
                tvWifiApCount.setText(String.format("Access Points: %d", readings.size()));
            }

            if (tvWifiVariance != null) {
                String varianceColor = variance < 5.0f ? "🟢" : "🔴";
                tvWifiVariance.setText(String.format("%s Variance: %.1f dBm²", varianceColor, variance));
            }

            if (tvWifiStatus != null) {
                if (variance < 5.0f) {
                    tvWifiStatus.setText("Status: Stable (Stationary)");
                } else {
                    tvWifiStatus.setText("Status: Changing (Moving)");
                }
            }

            if (tvWifiStrongestAp != null && !readings.isEmpty()) {
                com.example.wytv2.wifi.WiFiReading strongest = readings.get(0);
                for (com.example.wytv2.wifi.WiFiReading reading : readings) {
                    if (reading.rssi > strongest.rssi) {
                        strongest = reading;
                    }
                }

                String ssid = strongest.ssid != null && !strongest.ssid.isEmpty() ? strongest.ssid : "Hidden";
                tvWifiStrongestAp.setText(String.format("Strongest: %s (%d dBm)",
                        ssid, strongest.rssi));
            }
        });
    }

    public void updatePositionPanel(com.example.wytv2.localization.Position fusedPosition) {
        runOnUiThread(() -> {
            if (tvPositionStatus != null) {
                tvPositionStatus.setText("Status: Active ✓");
            }
            if (tvPositionCoordinates != null) {
                tvPositionCoordinates.setText(String.format(
                        "Position: (%.2f, %.2f) Floor %d",
                        fusedPosition.x, fusedPosition.y, fusedPosition.floor));
            }
            if (tvPositionHeading != null && stepDetectionService != null) {
                float heading = stepDetectionService.getCurrentHeading();
                tvPositionHeading.setText(String.format(
                        "Heading: %.0f°", Math.toDegrees(heading)));
            }
            if (tvPositionConfidence != null) {
                tvPositionConfidence.setText(String.format(
                        "Confidence: %.2f", fusedPosition.confidence));
            }

            // Update particle filter tester if active
            if (particleFilterTester != null && particleFilterTester.isTracking()) {
                float particleSpread = 0;
                float neff = 0;

                if (stepDetectionService != null) {
                    com.example.wytv2.localization.ParticleFilterLocalization pf = stepDetectionService
                            .getParticleFilter();
                    if (pf != null) {
                        particleSpread = pf.calculateParticleSpread();
                        neff = pf.calculateEffectiveSampleSize();
                    }
                }

                particleFilterTester.recordStep(fusedPosition, particleSpread, neff);

                if (tvPfTestStatus != null) {
                    tvPfTestStatus.setText(String.format(
                            "Status: Testing (Step %d/11)",
                            particleFilterTester.getStepCount()));
                }
                if (tvPfMeanError != null) {
                    float meanError = particleFilterTester.getCurrentMeanError();
                    tvPfMeanError.setText(String.format(
                            "Mean Error: %.2f m", meanError));
                }
            }

            // Update PF debug status
            if (stepDetectionService != null && tvPfDebug != null) {
                com.example.wytv2.localization.ParticleFilterLocalization pf = stepDetectionService.getParticleFilter();
                if (pf != null) {
                    tvPfDebug.setText(pf.getDebugStatus());
                }
            }
        });
    }

    private void initServiceListener() {
        settingsListener = new StepDetectionService.DeviceStateListener() {
            @Override
            public void onStepDetected(long timestamp, float stepLength, int totalSteps, int sessionSteps) {
                Log.d("SettingsActivity", "onStepDetected called: total=" + totalSteps + ", session=" + sessionSteps);
                runOnUiThread(() -> {
                    updateStepDisplay(totalSteps, sessionSteps);
                    updateRetrainingProgress(); // Update retraining progress in real-time
                });
            }

            @Override
            public void onDeviceStateChanged(boolean isStationary, float confidence) {
                runOnUiThread(() -> {
                    String stateText = isStationary ? "Stationary (Stable)" : "Moving";
                    updateDeviceState(stateText);
                    Log.d("SettingsActivity", "Device state changed: " + stateText);
                });
            }

            @Override
            public void onStepsReset() {
                runOnUiThread(() -> updateStepDisplay(0, 0));
            }

            @Override
            public void onInitialPositionEstimated(float x, float y, int floor) {
                // Optional log
            }

            @Override
            public void onCalibrationNodeMatched(float x, float y, String nodeId) {
                // Optional log
            }

            @Override
            public void onMagneticDataCollected(float[] magneticVector) {
                // Optional log
            }

            @Override
            public void onActivityRecognized(String activity, float confidence) {
                runOnUiThread(() -> {
                    if (tvMlActivityPrediction != null) {
                        tvMlActivityPrediction.setText(String.format("Prediction: %s (%.0f%%)",
                                activity, confidence * 100));
                    }
                });
            }

            @Override
            public void onWiFiRSSIUpdate(java.util.List<com.example.wytv2.wifi.WiFiReading> readings, float variance) {
                updateWiFiPanel(readings, variance);
            }

            @Override
            public void onPositionUpdated(com.example.wytv2.localization.Position fusedPosition) {
                updatePositionPanel(fusedPosition);
            }
        };
    }
}
