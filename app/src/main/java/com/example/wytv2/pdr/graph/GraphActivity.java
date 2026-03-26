//package com.example.wytv2.pdr.graph;
//
//import androidx.appcompat.app.AppCompatActivity;
//import android.content.Intent;
//import android.os.Bundle;
//import android.os.Handler;
//import android.widget.Button;
//import android.widget.CheckBox;
//import android.widget.SeekBar;
//import android.widget.TextView;
//import android.widget.Toast;
//
//import com.example.wytv2.R;
//import com.example.wytv2.pdr.StepDetectionService;
//
//public class GraphActivity extends AppCompatActivity
//        implements StepDetectionService.DeviceStateListener {
//
//    private StepDetectionService stepDetection;
//    private GraphView graphView;
//    private TextView thresholdValueText;
//    private SeekBar thresholdSeekBar;
//    private CheckBox showThresholdCheckBox;
//    private CheckBox showPointsCheckBox;
//    private Button clearButton;
//    private Button backButton;
//
//    private Handler graphUpdateHandler = new Handler();
//    private Runnable graphUpdateRunnable;
//    private boolean isRunning = true;
//
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_graph);
//
//        initializeViews();
//        setupStepDetection();
//        startGraphUpdates();
//
//        Toast.makeText(this,
//                "Graph shows real-time acceleration.\n" +
//                        "Adjust threshold to find optimal value for step detection.",
//                Toast.LENGTH_LONG).show();
//    }
//
//    private void initializeViews() {
//        // Graph view
//        graphView = findViewById(R.id.graph_view);
//
//        // Threshold controls
//        thresholdValueText = findViewById(R.id.threshold_value_text);
//        thresholdSeekBar = findViewById(R.id.threshold_seekbar);
//        showThresholdCheckBox = findViewById(R.id.show_threshold_checkbox);
//        showPointsCheckBox = findViewById(R.id.show_points_checkbox);
//
//        // Buttons
//        clearButton = findViewById(R.id.clear_button);
//        backButton = findViewById(R.id.back_button);
//
//        // Setup listeners
//        thresholdSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
//            @Override
//            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
//                float threshold = progress / 100f * 5f; // 0 to 5 m/s²
//                thresholdValueText.setText(String.format("Threshold: %.2f m/s²", threshold));
//                graphView.setThreshold(threshold);
//                if (stepDetection != null) {
//                    stepDetection.setBinaryThreshold(threshold);
//                }
//            }
//
//            @Override
//            public void onStartTrackingTouch(SeekBar seekBar) {}
//
//            @Override
//            public void onStopTrackingTouch(SeekBar seekBar) {}
//        });
//
//        showThresholdCheckBox.setOnCheckedChangeListener((buttonView, isChecked) -> {
//            graphView.setShowThreshold(isChecked);
//        });
//
//        showPointsCheckBox.setOnCheckedChangeListener((buttonView, isChecked) -> {
//            graphView.setShowPoints(isChecked);
//        });
//
//        clearButton.setOnClickListener(v -> {
//            graphView.clearData();
//            Toast.makeText(this, "Graph cleared", Toast.LENGTH_SHORT).show();
//        });
//
//        backButton.setOnClickListener(v -> {
//            finish();
//        });
//
//        // Set initial values
//        thresholdSeekBar.setProgress(24); // 1.2 m/s² (24% of 5)
//        showThresholdCheckBox.setChecked(true);
//        showPointsCheckBox.setChecked(false);
//    }
//
//    private void setupStepDetection() {
//        stepDetection = new StepDetectionService(this);
//        stepDetection.setDeviceStateListener(this);
//        stepDetection.start();
//
//        // Set initial threshold
//        float initialThreshold = 1.2f;
//        stepDetection.setBinaryThreshold(initialThreshold);
//        graphView.setThreshold(initialThreshold);
//    }
//
//    private void startGraphUpdates() {
//        graphUpdateRunnable = new Runnable() {
//            @Override
//            public void run() {
//                if (stepDetection != null) {
//                    // Get latest acceleration from step detection service
//                    float accel = stepDetection.getLatestAcceleration();
//                    long timestamp = System.currentTimeMillis();
//
//                    // Add to graph
//                    graphView.addDataPoint(accel, timestamp);
//                }
//
//                if (isRunning) {
//                    graphUpdateHandler.postDelayed(this, 20); // Update at ~50Hz
//                }
//            }
//        };
//        graphUpdateHandler.post(graphUpdateRunnable);
//    }
//
//    @Override
//    public void onDeviceStateChanged(boolean isStationary, float confidence) {
//        runOnUiThread(() -> {
//            // Update UI with state info if needed
//            String state = isStationary ? "STATIONARY" : "MOVING";
//            Toast.makeText(this, "State: " + state, Toast.LENGTH_SHORT).show();
//        });
//    }
//
//    @Override
//    public void onStepDetected(long timestamp, float stepLength,
//                               int totalSteps, int sessionSteps) {
//        runOnUiThread(() -> {
//            // Flash the graph when step is detected
//            graphView.setBackgroundColor(0x5500FF00); // Green flash
//            graphView.postDelayed(() -> graphView.setBackgroundColor(0xFF000000), 100);
//
//            Toast.makeText(this,
//                    String.format("Step detected! (Length: %.2fm)", stepLength),
//                    Toast.LENGTH_SHORT).show();
//        });
//    }
//
//    @Override
//    public void onStepsReset() {
//        // Not needed for graph
//    }
//
//    @Override
//    public void onInitialPositionEstimated(float x, float y, int floor) {
//
//    }
//
//    @Override
//    public void onCalibrationNodeMatched(float x, float y, String nodeId) {
//
//    }
//
//    @Override
//    public void onMagneticDataCollected(float[] magneticVector) {
//
//    }
//
//    @Override
//    protected void onResume() {
//        super.onResume();
//        isRunning = true;
//        if (stepDetection != null) {
//            stepDetection.start();
//        }
//        startGraphUpdates();
//    }
//
//    @Override
//    protected void onPause() {
//        super.onPause();
//        isRunning = false;
//        if (stepDetection != null) {
//            stepDetection.stop();
//        }
//        graphUpdateHandler.removeCallbacks(graphUpdateRunnable);
//    }
//
//    @Override
//    protected void onDestroy() {
//        super.onDestroy();
//        isRunning = false;
//        if (stepDetection != null) {
//            stepDetection.stop();
//        }
//        graphUpdateHandler.removeCallbacksAndMessages(null);
//    }
//}