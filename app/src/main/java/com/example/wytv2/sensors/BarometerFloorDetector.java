package com.example.wytv2.sensors;

import android.app.AlertDialog;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;

import java.util.ArrayDeque;
import java.util.Locale;

/**
 * Self-contained barometer-based floor detector.
 *
 * ISOLATION NOTE: This class is intentionally self-contained. To remove the feature:
 *   1. Delete this file.
 *   2. Remove the single call in MainActivity: initBarometerDetector().
 *   3. Remove the "📊 Barometer" button from activity_main.xml.
 *
 * ---- Tuning parameters (adjust via the debug dialog) ----
 *   PRESSURE_PER_FLOOR_HPA  ~0.3 hPa for a 3m floor (0.12 hPa/m × 3m ≈ 0.36)
 *                            Tune upward if too many false positives.
 *   SMOOTHING_SAMPLES       Window size for moving-average noise reduction.
 *   FLOOR_HYSTERESIS        Fraction of one floor needed before a change fires.
 *
 * Calibration physics:
 *   Standard atmosphere ≈ 0.12 hPa/m at sea level.
 *   Pressure DECREASES when you go UP (lower altitude = higher pressure).
 *   delta = baseline − smoothed  →  positive = ascended, negative = descended.
 */
public class BarometerFloorDetector implements SensorEventListener {

    private static final String TAG = "Barometer";

    // ---- Tuning parameters (mutable at runtime via debug dialog) ----
    public float pressurePerFloorHpa = 0.30f;  // hPa ≈ 3m floor height
    public int   smoothingSamples    = 15;      // moving-average window
    public float floorHysteresis     = 0.65f;  // fraction of floor before event fires

    // ---- Sensor hardware ----
    private final SensorManager sensorManager;
    private final Sensor pressureSensor;

    // ---- State ----
    private final ArrayDeque<Float> buffer = new ArrayDeque<>();
    private float baselinePressure = Float.NaN;
    private float lastSmoothed     = Float.NaN;
    private int   currentFloor     = 0;

    // ---- Callback ----
    public interface Listener {
        /** Called on every new smoothed pressure reading (on UI thread). */
        void onPressureUpdate(float smoothed, float baseline, float deltaHpa, int estimatedFloor);
        /** Called when a confirmed floor change is detected. */
        void onFloorChanged(int newFloor, int delta, String direction);
    }

    private Listener listener;
    private final Handler uiHandler = new Handler(Looper.getMainLooper());

    // ---- Constructor ----
    public BarometerFloorDetector(Context context) {
        sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        pressureSensor = (sensorManager != null)
                ? sensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE) : null;
    }

    public boolean isAvailable() { return pressureSensor != null; }
    public void setListener(Listener l) { this.listener = l; }
    public int  getCurrentFloor()       { return currentFloor; }
    public float getBaselinePressure()  { return baselinePressure; }
    public float getLastSmoothed()      { return lastSmoothed; }

    public void start() {
        if (pressureSensor != null) {
            sensorManager.registerListener(this, pressureSensor,
                    SensorManager.SENSOR_DELAY_NORMAL);
            Log.d(TAG, "Barometer started");
        } else {
            Log.w(TAG, "No barometer sensor available on this device");
        }
    }

    public void stop() {
        if (sensorManager != null) sensorManager.unregisterListener(this);
    }

    /**
     * Set the current smoothed pressure as the floor-0 baseline.
     * Call this when the user is standing on a known reference floor.
     */
    public void calibrate() {
        if (!Float.isNaN(lastSmoothed)) {
            baselinePressure = lastSmoothed;
            currentFloor = 0;
            Log.d(TAG, String.format("Calibrated: baseline=%.3f hPa", baselinePressure));
        }
    }

    // ---- SensorEventListener ----

    @Override
    public void onSensorChanged(SensorEvent event) {
        float raw = event.values[0];

        // Moving average smoothing
        buffer.addLast(raw);
        if (buffer.size() > smoothingSamples) buffer.removeFirst();
        float smoothed = 0;
        for (float v : buffer) smoothed += v;
        smoothed /= buffer.size();
        lastSmoothed = smoothed;

        // Auto-calibrate on first reading if not yet set
        if (Float.isNaN(baselinePressure)) {
            baselinePressure = smoothed;
            Log.d(TAG, String.format("Auto-calibrated baseline: %.3f hPa", baselinePressure));
        }

        // Compute altitude delta: positive = ascended (pressure dropped)
        final float delta = baselinePressure - smoothed;
        final int estimated = Math.round(delta / pressurePerFloorHpa);

        // Notify listener on UI thread
        if (listener != null) {
            final float snap = smoothed;
            uiHandler.post(() ->
                    listener.onPressureUpdate(snap, baselinePressure, delta, estimated));
        }

        // Hysteresis check: require at least `floorHysteresis` of a floor change
        if (estimated != currentFloor) {
            float excess = Math.abs(delta - currentFloor * pressurePerFloorHpa);
            if (excess >= floorHysteresis * pressurePerFloorHpa) {
                int floorDelta = estimated - currentFloor;
                String dir = floorDelta > 0 ? "UP ↑" : "DOWN ↓";
                Log.d(TAG, String.format(
                        "Floor change: %d → %d (%s)  Δ=%.3f hPa  baseline=%.3f",
                        currentFloor, estimated, dir, delta, baselinePressure));
                currentFloor = estimated;
                if (listener != null) {
                    final int fd = floorDelta, fl = currentFloor;
                    final String fdir = dir;
                    uiHandler.post(() -> listener.onFloorChanged(fl, fd, fdir));
                }
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) { }

    // =========================================================================
    // Debug dialog — a standalone AlertDialog showing live readings + tuning
    // =========================================================================

    /**
     * Show a live debug dialog for barometer readings and tuning.
     * Everything is self-contained here; no layout XML needed.
     *
     * @param context activity context
     */
    public void showDebugDialog(Context context) {
        LinearLayout root = new LinearLayout(context);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = dpToPx(context, 16);
        root.setPadding(pad, pad, pad, pad);

        // Status label (updated by listener override)
        TextView tvStatus = new TextView(context);
        tvStatus.setTextSize(14);
        tvStatus.setText(formatStatus());
        tvStatus.setPadding(0, 0, 0, dpToPx(context, 12));
        root.addView(tvStatus);

        // ---- Calibrate button ----
        Button btnCal = new Button(context);
        btnCal.setText("📍 Calibrate (set floor 0)");
        btnCal.setOnClickListener(v -> {
            calibrate();
            tvStatus.setText(formatStatus());
        });
        root.addView(btnCal);

        addSpacer(root, context, 8);

        // ---- Pressure per floor seekbar: 0.10 – 0.80 hPa (step 0.01) ----
        TextView tvFloorLabel = new TextView(context);
        tvFloorLabel.setText(String.format(Locale.US,
                "Pressure/floor: %.2f hPa  (≈%.0fm)", pressurePerFloorHpa,
                pressurePerFloorHpa / 0.12f));
        root.addView(tvFloorLabel);

        SeekBar sbFloor = new SeekBar(context);
        sbFloor.setMax(70); // 0.10 + progress*0.01
        sbFloor.setProgress(Math.round((pressurePerFloorHpa - 0.10f) / 0.01f));
        sbFloor.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar s, int p, boolean u) {
                pressurePerFloorHpa = 0.10f + p * 0.01f;
                tvFloorLabel.setText(String.format(Locale.US,
                        "Pressure/floor: %.2f hPa  (≈%.0fm)", pressurePerFloorHpa,
                        pressurePerFloorHpa / 0.12f));
            }
            public void onStartTrackingTouch(SeekBar s) {}
            public void onStopTrackingTouch(SeekBar s) {}
        });
        root.addView(sbFloor);

        addSpacer(root, context, 6);

        // ---- Hysteresis seekbar: 0.40 – 0.95 ----
        TextView tvHystLabel = new TextView(context);
        tvHystLabel.setText(String.format(Locale.US,
                "Hysteresis: %.0f%%  (sensitivity)", floorHysteresis * 100));
        root.addView(tvHystLabel);

        SeekBar sbHyst = new SeekBar(context);
        sbHyst.setMax(55); // 0.40 + progress*0.01
        sbHyst.setProgress(Math.round((floorHysteresis - 0.40f) / 0.01f));
        sbHyst.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar s, int p, boolean u) {
                floorHysteresis = 0.40f + p * 0.01f;
                tvHystLabel.setText(String.format(Locale.US,
                        "Hysteresis: %.0f%%  (sensitivity)", floorHysteresis * 100));
            }
            public void onStartTrackingTouch(SeekBar s) {}
            public void onStopTrackingTouch(SeekBar s) {}
        });
        root.addView(sbHyst);

        addSpacer(root, context, 6);

        // ---- Smoothing seekbar: 5 – 30 samples ----
        TextView tvSmoothLabel = new TextView(context);
        tvSmoothLabel.setText("Smoothing: " + smoothingSamples + " samples");
        root.addView(tvSmoothLabel);

        SeekBar sbSmooth = new SeekBar(context);
        sbSmooth.setMax(25); // 5 + progress
        sbSmooth.setProgress(smoothingSamples - 5);
        sbSmooth.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar s, int p, boolean u) {
                smoothingSamples = 5 + p;
                tvSmoothLabel.setText("Smoothing: " + smoothingSamples + " samples");
            }
            public void onStartTrackingTouch(SeekBar s) {}
            public void onStopTrackingTouch(SeekBar s) {}
        });
        root.addView(sbSmooth);

        addSpacer(root, context, 10);

        // Live readings — update every second while dialog is open
        TextView tvLive = new TextView(context);
        tvLive.setTextSize(13);
        tvLive.setText(formatStatus());
        root.addView(tvLive);

        // Temp override listener to update the live label
        Listener original = listener;
        setListener(new Listener() {
            @Override
            public void onPressureUpdate(float s, float b, float d, int ef) {
                tvLive.setText(String.format(Locale.US,
                        "Pressure: %.3f hPa\nBaseline: %.3f hPa\nΔ:        %+.3f hPa\nEst. floor: %+d",
                        s, b, d, ef));
                if (original != null) original.onPressureUpdate(s, b, d, ef);
            }
            @Override
            public void onFloorChanged(int fl, int fd, String dir) {
                tvStatus.setText("⚠ Floor change: " + dir + " → floor " + fl);
                if (original != null) original.onFloorChanged(fl, fd, dir);
            }
        });

        AlertDialog dialog = new AlertDialog.Builder(context)
                .setTitle("📊 Barometer Floor Detector")
                .setView(root)
                .setPositiveButton("Close", null)
                .setOnDismissListener(d -> setListener(original)) // restore original listener
                .create();
        dialog.show();
    }

    // ---- Helpers ----

    private String formatStatus() {
        if (Float.isNaN(lastSmoothed)) return "Waiting for sensor…";
        float delta = Float.isNaN(baselinePressure) ? 0 : baselinePressure - lastSmoothed;
        return String.format(Locale.US,
                "Pressure: %.3f hPa\nBaseline: %.3f hPa\nΔ: %+.3f hPa\nCurrent floor: %+d",
                lastSmoothed, Float.isNaN(baselinePressure) ? 0f : baselinePressure, delta, currentFloor);
    }

    private static void addSpacer(LinearLayout parent, Context ctx, int dp) {
        View v = new View(ctx);
        v.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(ctx, dp)));
        parent.addView(v);
    }

    private static int dpToPx(Context ctx, int dp) {
        return Math.round(dp * ctx.getResources().getDisplayMetrics().density);
    }
}
