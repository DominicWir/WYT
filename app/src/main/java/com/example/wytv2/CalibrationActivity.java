package com.example.wytv2;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Bundle;
import android.os.IBinder;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.wytv2.pdr.StepDetectionService;
import com.example.wytv2.pdr.calibration.ThresholdCalibrator;

/**
 * Activity for calibrating step detection threshold.
 * Users walk a known number of steps and the app automatically adjusts the
 * threshold.
 */
public class CalibrationActivity extends AppCompatActivity {

    private EditText actualStepsInput;
    private TextView statusText;
    private TextView detectedStepsText;
    private TextView thresholdText;
    private TextView accuracyText;
    private TextView recommendationText;
    private LinearLayout accuracyLayout;
    private Button startButton;
    private Button stopButton;
    private Button saveButton;

    private StepDetectionService stepService;
    private boolean serviceBound = false;
    private boolean isCalibrating = false;

    private int detectedSteps = 0;
    private float currentThreshold = 1.0f;
    private float calibratedThreshold = 1.0f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_calibration);

        // Initialize UI elements
        actualStepsInput = findViewById(R.id.actual_steps_input);
        statusText = findViewById(R.id.status_text);
        detectedStepsText = findViewById(R.id.detected_steps_text);
        thresholdText = findViewById(R.id.threshold_text);
        accuracyText = findViewById(R.id.accuracy_text);
        recommendationText = findViewById(R.id.recommendation_text);
        accuracyLayout = findViewById(R.id.accuracy_layout);
        startButton = findViewById(R.id.start_button);
        stopButton = findViewById(R.id.stop_button);
        saveButton = findViewById(R.id.save_button);

        // Set up button listeners
        startButton.setOnClickListener(v -> startCalibration());
        stopButton.setOnClickListener(v -> stopCalibration());
        saveButton.setOnClickListener(v -> saveCalibration());

        // Get service instance from MainActivity
        stepService = MainActivity.getStepDetectionService();
        if (stepService != null) {
            serviceBound = true;
            currentThreshold = stepService.getBinaryThreshold();
            thresholdText.setText(String.format("%.2f", currentThreshold));
        } else {
            Toast.makeText(this, "Service not available", Toast.LENGTH_SHORT).show();
        }
    }

    private void startCalibration() {
        if (!serviceBound) {
            Toast.makeText(this, "Service not ready", Toast.LENGTH_SHORT).show();
            return;
        }

        String input = actualStepsInput.getText().toString();
        if (input.isEmpty()) {
            Toast.makeText(this, "Please enter target steps", Toast.LENGTH_SHORT).show();
            return;
        }

        isCalibrating = true;
        detectedSteps = 0;

        // Update UI
        startButton.setEnabled(false);
        stopButton.setEnabled(true);
        saveButton.setEnabled(false);
        accuracyLayout.setVisibility(View.GONE);
        recommendationText.setVisibility(View.GONE);
        statusText.setText("Calibrating... Start walking!");
        detectedStepsText.setText("0");

        // Start calibration in service
        stepService.startCalibration(new StepDetectionService.CalibrationListener() {
            @Override
            public void onCalibrationStepDetected(int stepCount, float threshold) {
                runOnUiThread(() -> {
                    detectedSteps = stepCount;
                    currentThreshold = threshold;
                    detectedStepsText.setText(String.valueOf(stepCount));
                    thresholdText.setText(String.format("%.2f", threshold));
                });
            }
        });
    }

    private void stopCalibration() {
        if (!serviceBound || !isCalibrating) {
            return;
        }

        isCalibrating = false;

        // Stop calibration in service
        detectedSteps = stepService.stopCalibration();

        // Get actual steps from input
        int actualSteps = Integer.parseInt(actualStepsInput.getText().toString());

        // Calculate calibrated threshold
        float oldThreshold = currentThreshold;
        calibratedThreshold = ThresholdCalibrator.calculateOptimalThreshold(
                actualSteps,
                detectedSteps,
                currentThreshold);

        // AUTO-APPLY new threshold immediately (don't wait for save)
        stepService.setThreshold(calibratedThreshold);
        currentThreshold = calibratedThreshold;

        // Calculate accuracy
        float accuracy = ThresholdCalibrator.calculateAccuracy(actualSteps, detectedSteps);
        String recommendation = ThresholdCalibrator.getCalibrationRecommendation(actualSteps, detectedSteps);

        // Update UI
        startButton.setEnabled(true);
        stopButton.setEnabled(false);
        saveButton.setEnabled(true);
        accuracyLayout.setVisibility(View.VISIBLE);
        recommendationText.setVisibility(View.VISIBLE);

        statusText.setText("Threshold auto-applied! Try again or save.");
        thresholdText.setText(String.format("%.2f → %.2f", oldThreshold, calibratedThreshold));
        accuracyText.setText(String.format("%.1f%%", accuracy));
        recommendationText.setText(recommendation + "\n\nCalibrate again for better accuracy or tap Save.");

        // Show result toast
        if (accuracy >= 90) {
            Toast.makeText(this, "Excellent calibration!", Toast.LENGTH_SHORT).show();
        } else if (accuracy >= 80) {
            Toast.makeText(this, "Good calibration. You can recalibrate for better accuracy.", Toast.LENGTH_LONG)
                    .show();
        } else {
            Toast.makeText(this, "Poor calibration. Please try again with more careful counting.", Toast.LENGTH_LONG)
                    .show();
        }
    }

    private void saveCalibration() {
        if (!serviceBound) {
            return;
        }

        // Save calibrated threshold to service
        stepService.saveCalibratedThreshold(calibratedThreshold);

        Toast.makeText(this, "Calibration saved!", Toast.LENGTH_SHORT).show();

        // Return to main activity
        finish();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        // Stop calibration if still running
        if (isCalibrating && serviceBound && stepService != null) {
            stepService.stopCalibration();
        }
    }
}
