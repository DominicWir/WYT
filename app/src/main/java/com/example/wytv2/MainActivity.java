package com.example.wytv2;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Context;
import android.content.pm.PackageManager;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.wytv2.mapcontext.BuildingMap;
import com.example.wytv2.mapcontext.BuildingMapRepository;
import com.example.wytv2.mapcontext.FloorRecord;
import com.example.wytv2.pdr.StepDetectionService;
import com.example.wytv2.sensors.BarometerFloorDetector;
import com.example.wytv2.zones.ZoneAlert;
import com.example.wytv2.zones.ZoneAlertRepository;
import com.example.wytv2.zones.ZoneMarker;
import com.example.wytv2.zones.ZoneMarkerRepository;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class MainActivity extends AppCompatActivity {

    private static MainActivity instance;
    private StepDetectionService stepDetectionService;
    private StepDetectionService.DeviceStateListener mainActivityListener;

    public static StepDetectionService getStepDetectionService() {
        if (instance != null) {
            return instance.stepDetectionService;
        }
        return null;
    }

    // UI Elements
    private TextView tvState;
    private TextView tvSessionSteps;
    private SeekBar thresholdSeekBar;
    private Button btnResetSteps;

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

    // Map View UI
    private com.example.wytv2.visualization.LocationMapView locationMapView;
    private TextView tvPositionCompact;
    private TextView tvHeadingCompact;
    private TextView tvCurrentDeviation;

    private Button btnClearPath;
    private Button btnCenterView;

    // Zone markers
    private ZoneMarkerRepository zoneRepo;
    private TextView tvCurrentZone;
    private float currentPdrX = 0f;
    private float currentPdrY = 0f;
    private final Handler zoneHandler = new Handler(Looper.getMainLooper());
    private static final float ZONE_RADIUS_M = 3.0f;

    // Zone alerts
    private ZoneAlertRepository alertRepo;
    private String currentZoneMarkerId = null; // zone currently inside (null = none)
    private final Set<String> snoozedAlertIds = new HashSet<>(); // reset on zone exit
    private boolean alertDialogShowing = false;

    // Menu panel
    private LinearLayout menuPanel;
    private boolean menuOpen = false;

    // 15-second inactivity auto-close timer
    private final Handler inactivityHandler = new Handler(Looper.getMainLooper());
    private static final long INACTIVITY_TIMEOUT_MS = 15_000L;
    private final Runnable inactivityRunnable = () -> {
        closeMenu();
        android.util.Log.d("MainActivity", "Edit menu closed after 15 s of inactivity");
    };

    // Barometer floor detection
    private BarometerFloorDetector barometerDetector;

    // ---- Building + floor map context ----
    private BuildingMapRepository buildingMapRepo;
    private BuildingMap currentBuilding;
    private int currentMapFloor = 0;
    /** SharedPreferences key for persisting the last-known floor across sessions. */
    private static final String PREF_MAP_CONTEXT = "map_context";
    private static final String PREF_LAST_FLOOR  = "last_floor";
    private static final int    PREF_NO_FLOOR    = Integer.MIN_VALUE;
    /** Stores the {x,y} of the last zone the user was in on each floor before leaving. */
    private final java.util.Map<Integer, float[]> lastZonePerFloor = new java.util.HashMap<>();

    private static final int PERMISSION_REQUEST_CODE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Store instance for static access
        instance = this;



        // Request necessary permissions
        requestPermissions();

        // Initialize UI with error handling
        try {
            initUI();
        } catch (Exception e) {
            Log.e("MainActivity", "Error initializing UI", e);
        }

        // Initialize Step Detection Service with error handling
        try {
            initStepDetectionService();
        } catch (Exception e) {
            Log.e("MainActivity", "Error initializing step detection", e);
        }

        // Identify building (WiFi SSID) and load the correct floor map
        // Must run after initStepDetectionService and initBarometerDetector (called from initUI)
        initMapContext();
    }



    private void requestPermissions() {
        String[] permissions = {
                Manifest.permission.ACTIVITY_RECOGNITION,
                Manifest.permission.ACCESS_FINE_LOCATION
        };

        boolean allGranted = true;
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                allGranted = false;
                break;
            }
        }

        if (!allGranted) {
            ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQUEST_CODE);
        }
    }

    private void initUI() {
        tvState = findViewById(R.id.device_state);

        // Debug panel views are not present in the current layout — all null-guarded below.
        btnResetSteps = null;
        tvMlBufferStatus = null;
        tvMlActivityPrediction = null;
        tvMlThresholdStatus = null;
        tvMlLastUpdate = null;
        tvWifiApCount = null;
        tvWifiVariance = null;
        tvWifiStatus = null;
        tvWifiStrongestAp = null;

        // Map View
        locationMapView = findViewById(R.id.location_map_view);
        tvPositionCompact = findViewById(R.id.tv_position_compact);
        tvHeadingCompact = findViewById(R.id.tv_heading_compact);
        tvCurrentDeviation = findViewById(R.id.tv_current_deviation);

        btnClearPath = findViewById(R.id.btn_clear_path);
        btnCenterView = findViewById(R.id.btn_center_view);

        // Initialize threshold value
        float currentThreshold = 1.5f;
        updateThresholdDisplay(currentThreshold);


        // Setup threshold seekbar with null check
        if (thresholdSeekBar != null) {
            thresholdSeekBar.setMax(30); // 0.9 to 2.4 in 0.05 increments
            thresholdSeekBar.setProgress(6); // Default: 1.2 (0.9 + 6*0.05)
            updateThresholdDisplay(1.2f);

            thresholdSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    float threshold = 0.9f + (progress * 0.05f);
                    updateThresholdDisplay(threshold);
                    if (stepDetectionService != null) {
                        stepDetectionService.setBinaryThreshold(threshold);
                    }
                }

                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {
                }

                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {
                }
            });
        }

        // Reset steps button with null check
        if (btnResetSteps != null) {
            btnResetSteps.setOnClickListener(v -> {
                if (stepDetectionService != null) {
                    stepDetectionService.resetAllSteps();
                    updateStepDisplay(0, 0, 0.0f);
                }
            });
        }


        // Map view button handlers (inside the FAB menu panel)
        btnClearPath = findViewById(R.id.btn_clear_path);
        btnCenterView = findViewById(R.id.btn_center_view);

        if (btnCenterView != null)
            btnCenterView.setOnClickListener(v -> {
                resetInactivityTimer();
                if (locationMapView != null) locationMapView.centerView();
            });
        if (btnClearPath != null)
            btnClearPath.setOnClickListener(v -> {
                resetInactivityTimer();
                if (locationMapView != null) locationMapView.clearPath();
            });

        Button btnTogglePath = findViewById(R.id.btn_toggle_path);
        if (btnTogglePath != null) {
            btnTogglePath.setOnClickListener(v -> {
                resetInactivityTimer();
                if (locationMapView != null) {
                    boolean nowVisible = !locationMapView.isPathVisible();
                    locationMapView.setPathVisible(nowVisible);
                    btnTogglePath.setText(nowVisible ? "〰 Path: ON" : "〰 Path: OFF");
                    btnTogglePath.setBackgroundTintList(
                            android.content.res.ColorStateList.valueOf(
                                    nowVisible ? 0xFF2E7D32 : 0xFF424242));
                }
            });
        }

        // FAB menu toggle
        menuPanel = findViewById(R.id.menu_panel);
        android.widget.ImageButton fabMenu = findViewById(R.id.fab_menu);
        if (fabMenu != null) {
            fabMenu.setOnClickListener(v -> { if (menuOpen) closeMenu(); else openMenu(); });
        }

        // ---- Edit Map sub-panel ----
        View editMapPanel = findViewById(R.id.edit_map_panel);
        Button btnEditMap = findViewById(R.id.btn_edit_map);
        if (btnEditMap != null) {
            btnEditMap.setOnClickListener(v -> {
                resetInactivityTimer();
                if (editMapPanel == null) return;
                boolean open = editMapPanel.getVisibility() == View.VISIBLE;
                if (open) {
                    editMapPanel.setVisibility(View.GONE);
                    exitEditMode();
                    btnEditMap.setText("✏️ Edit Map");
                    btnEditMap.setBackgroundTintList(
                            android.content.res.ColorStateList.valueOf(0xFF1565C0));
                } else {
                    editMapPanel.setVisibility(View.VISIBLE);
                    enterEditMode();
                    btnEditMap.setText("✅ Done");
                    btnEditMap.setBackgroundTintList(
                            android.content.res.ColorStateList.valueOf(0xFF2E7D32));
                }
            });
        }

        // Zone system buttons (inside Edit Map sub-panel)
        tvCurrentZone = findViewById(R.id.tv_current_zone);

        Button btnSaveZone = findViewById(R.id.btn_save_zone);
        if (btnSaveZone != null)
            btnSaveZone.setOnClickListener(v -> { resetInactivityTimer(); showSaveZoneDialog(); });

        Button btnManageZones = findViewById(R.id.btn_manage_zones);
        if (btnManageZones != null)
            btnManageZones.setOnClickListener(v -> { resetInactivityTimer(); showManageZonesDialog(); });

        // Move Zone: pick a zone then drag it on the map
        Button btnMoveZone = findViewById(R.id.btn_move_zone);
        if (btnMoveZone != null) {
            btnMoveZone.setOnClickListener(v -> { resetInactivityTimer(); showMoveZoneDialog(); });
        }

        // Switch Floor: manual floor picker
        Button btnSwitchFloor = findViewById(R.id.btn_switch_floor);
        if (btnSwitchFloor != null) {
            btnSwitchFloor.setOnClickListener(v -> { resetInactivityTimer(); showSwitchFloorDialog(); });
        }

        // Calibrate Floor: re-identify from barometer
        Button btnCalibrateFloor = findViewById(R.id.btn_calibrate_floor);
        if (btnCalibrateFloor != null) {
            btnCalibrateFloor.setOnClickListener(v -> { resetInactivityTimer(); calibrateFloorFromBarometer(); });
        }

        // ---- Alerts button ----
        Button btnAlerts = findViewById(R.id.btn_alerts);
        if (btnAlerts != null)
            btnAlerts.setOnClickListener(v -> { showZoneAlertsDialog(); closeMenu(); });

        // ---- Menu close (×) button ----
        android.widget.ImageButton btnCloseMenu = findViewById(R.id.btn_close_menu);
        if (btnCloseMenu != null) {
            btnCloseMenu.setOnClickListener(v -> closeMenu());
        }

        // Zone identification runs continuously
        startZoneIdentification();

        initBarometerDetector();
    }

    // ---- Edit Map mode helpers ----
    private boolean isEditModeActive = false;

    private void enterEditMode() {
        isEditModeActive = true;
        if (stepDetectionService != null)
            stepDetectionService.setPositionCorrectionEnabled(false);
        Log.d("EditMode", "Edit mode ON — correction paused");
    }

    private void exitEditMode() {
        isEditModeActive = false;
        if (locationMapView != null) locationMapView.setDraggableZone(null, null);
        if (stepDetectionService != null)
            stepDetectionService.setPositionCorrectionEnabled(true);
        Log.d("EditMode", "Edit mode OFF — correction resumed");
    }

    /** Show a picker, then activate drag mode for the chosen zone. */
    private void showMoveZoneDialog() {
        if (zoneRepo == null) return;
        List<com.example.wytv2.zones.ZoneMarker> zones = zoneRepo.getAll();
        if (zones.isEmpty()) {
            android.widget.Toast.makeText(this, "No zones to move", android.widget.Toast.LENGTH_SHORT).show();
            return;
        }
        String[] names = zones.stream().map(z -> z.name).toArray(String[]::new);
        new android.app.AlertDialog.Builder(this)
                .setTitle("Select zone to move")
                .setItems(names, (d, which) -> {
                    com.example.wytv2.zones.ZoneMarker chosen = zones.get(which);
                    android.widget.Toast.makeText(this,
                            "Drag \"" + chosen.name + "\" to its new position, then tap Done",
                            android.widget.Toast.LENGTH_LONG).show();
                    if (locationMapView != null) {
                        locationMapView.setDraggableZone(chosen, (zone, newX, newY) -> {
                            zone.x = newX;
                            zone.y = newY;
                            if (zoneRepo != null) zoneRepo.update(zone);
                            refreshMapMarkers();
                            updateServiceZonePositions();
                            Log.d("EditMode", String.format(
                                    "Zone '%s' moved to (%.2f, %.2f)", zone.name, newX, newY));
                        });
                    }
                })
                .show();
    }

    /**
     * Show a picker listing all registered floors for the current building.
     * Selecting one calls switchToFloor() and persists the choice for next startup.
     */
    private void showSwitchFloorDialog() {
        if (currentBuilding == null || currentBuilding.floors == null
                || currentBuilding.floors.isEmpty()) {
            android.widget.Toast.makeText(this, "No floors registered for this building",
                    android.widget.Toast.LENGTH_SHORT).show();
            return;
        }
        // Sort floors by floor number
        java.util.List<FloorRecord> sorted = new java.util.ArrayList<>(currentBuilding.floors);
        sorted.sort(java.util.Comparator.comparingInt(f -> f.floorNumber));

        String[] labels = sorted.stream()
                .map(f -> BuildingMapRepository.floorLabel(f.floorNumber)
                        + (f.floorNumber == currentMapFloor ? " ✓" : ""))
                .toArray(String[]::new);

        new android.app.AlertDialog.Builder(this)
                .setTitle("Switch Floor")
                .setItems(labels, (d, which) -> {
                    int chosen = sorted.get(which).floorNumber;
                    if (chosen == currentMapFloor) return;
                    switchToFloor(chosen);
                    // Persist so next startup opens here directly
                    getSharedPreferences(PREF_MAP_CONTEXT, Context.MODE_PRIVATE)
                            .edit()
                            .putInt(PREF_LAST_FLOOR, chosen)
                            .apply();
                    Log.d("MapContext", "Manual floor switch → " + chosen);
                })
                .setNeutralButton("Delete Floor…", (d, w) -> showDeleteFloorDialog(sorted))
                .setNegativeButton("Cancel", null)
                .show();
    }

    /** Let the user delete a non-active floor from the building record. */
    private void showDeleteFloorDialog(java.util.List<FloorRecord> sorted) {
        if (currentBuilding == null) return;
        // Build labels, greying out the active floor (can't delete it)
        String[] labels = sorted.stream()
                .map(f -> BuildingMapRepository.floorLabel(f.floorNumber)
                        + (f.floorNumber == currentMapFloor ? " (current — cannot delete)" : ""))
                .toArray(String[]::new);

        new android.app.AlertDialog.Builder(this)
                .setTitle("Delete a Floor")
                .setItems(labels, (d, which) -> {
                    FloorRecord target = sorted.get(which);
                    if (target.floorNumber == currentMapFloor) {
                        android.widget.Toast.makeText(this,
                                "Cannot delete the active floor. Switch floors first.",
                                android.widget.Toast.LENGTH_SHORT).show();
                        return;
                    }
                    if (currentBuilding.floors.size() <= 1) {
                        android.widget.Toast.makeText(this,
                                "Cannot delete the only floor.",
                                android.widget.Toast.LENGTH_SHORT).show();
                        return;
                    }
                    String label = BuildingMapRepository.floorLabel(target.floorNumber);
                    new android.app.AlertDialog.Builder(this)
                            .setTitle("Delete " + label + "?")
                            .setMessage("This will permanently erase all zones and alerts on " + label + ".")
                            .setPositiveButton("Delete", (d2, w2) -> {
                                buildingMapRepo.deleteFloor(currentBuilding.id, target.floorNumber);
                                // Reload building record
                                currentBuilding = buildingMapRepo.findBySsid(currentBuilding.ssid);
                                android.widget.Toast.makeText(this,
                                        label + " deleted.", android.widget.Toast.LENGTH_SHORT).show();
                                Log.d("MapContext", "Floor deleted: " + target.floorNumber);
                            })
                            .setNegativeButton("Cancel", null)
                            .show();
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    /**
     * Re-identify floor from current barometer reading and switch if incorrect.
     * Also triggers an immediate P0→P3 position snap.
     */
    private void calibrateFloorFromBarometer() {
        if (barometerDetector == null || !barometerDetector.isAvailable()) {
            android.widget.Toast.makeText(this,
                    "Barometer unavailable on this device",
                    android.widget.Toast.LENGTH_SHORT).show();
            return;
        }
        if (currentBuilding == null) {
            android.widget.Toast.makeText(this, "No building loaded",
                    android.widget.Toast.LENGTH_SHORT).show();
            return;
        }
        float pressure = barometerDetector.getLastSmoothed();
        if (Float.isNaN(pressure)) {
            android.widget.Toast.makeText(this,
                    "No barometer reading yet — try again in a moment",
                    android.widget.Toast.LENGTH_SHORT).show();
            return;
        }
        int correctFloor = buildingMapRepo.identifyFloor(currentBuilding, pressure);
        String floorLabel = BuildingMapRepository.floorLabel(correctFloor);
        if (correctFloor != currentMapFloor) {
            Log.d("MapContext", String.format(
                    "Calibrate: pressure=%.3f hPa → floor %d (%s), was %d",
                    pressure, correctFloor, floorLabel, currentMapFloor));
            switchToFloor(correctFloor);
            getSharedPreferences(PREF_MAP_CONTEXT, Context.MODE_PRIVATE)
                    .edit().putInt(PREF_LAST_FLOOR, correctFloor).apply();
            android.widget.Toast.makeText(this,
                    "Floor corrected: " + floorLabel, android.widget.Toast.LENGTH_SHORT).show();
        } else {
            android.widget.Toast.makeText(this,
                    "Floor already correct (" + floorLabel + ")",
                    android.widget.Toast.LENGTH_SHORT).show();
        }
        // Always run position correction after a calibration
        if (stepDetectionService != null) {
            stepDetectionService.triggerStationaryCorrection();
        }
    }

    private void openMenu() {
        if (menuPanel != null) {
            menuPanel.setVisibility(View.VISIBLE);
            menuOpen = true;
            resetInactivityTimer();
        }
    }
    private void closeMenu() {
        inactivityHandler.removeCallbacks(inactivityRunnable);
        if (menuPanel != null) {
            // Also collapse edit sub-panel
            View editPanel = menuPanel.findViewById(R.id.edit_map_panel);
            if (editPanel != null) editPanel.setVisibility(View.GONE);
            Button btnEdit = menuPanel.findViewById(R.id.btn_edit_map);
            if (btnEdit != null) {
                btnEdit.setText("✏️ Edit Map");
                btnEdit.setBackgroundTintList(
                        android.content.res.ColorStateList.valueOf(0xFF1565C0));
            }
            exitEditMode();
            menuPanel.setVisibility(View.GONE);
            menuOpen = false;
        }
    }
    /** Restart the 15-second inactivity countdown. Call this on every menu interaction. */
    private void resetInactivityTimer() {
        inactivityHandler.removeCallbacks(inactivityRunnable);
        inactivityHandler.postDelayed(inactivityRunnable, INACTIVITY_TIMEOUT_MS);
    }

    // Barometer floor detection
    private void initBarometerDetector() {
        barometerDetector = new BarometerFloorDetector(this);
        if (!barometerDetector.isAvailable()) {
            android.util.Log.w("MainActivity", "No barometer on this device — feature disabled");
            return;
        }
        barometerDetector.setListener(new BarometerFloorDetector.Listener() {
            @Override
            public void onPressureUpdate(float s, float b, float d, int ef) {
                // Silent — visible only in debug dialog
            }
            @Override
            public void onFloorChanged(int newFloor, int delta, String direction) {
                handleBarometerFloorChange(newFloor, direction);
            }
        });
        barometerDetector.start();
    }

    // =========================================================================
    // Building + Floor Map Context
    //=========================================================================

    private void initMapContext() {
        buildingMapRepo = new BuildingMapRepository(this);
        String ssid = getCurrentSsid();
        Log.d("MapContext", "Detected WiFi SSID: " + ssid);

        BuildingMap building = buildingMapRepo.findBySsid(ssid);
        if (building != null) {
            currentBuilding = building;
            Log.d("MapContext", "Known building: " + building.buildingName);
            identifyStartingFloor();
        } else {
            Log.d("MapContext", "Unknown location — prompting user");
            showNewLocationDialog(ssid);
        }
    }

    /** Read connected WiFi SSID. Falls back to “unknown” if unavailable. */
    private String getCurrentSsid() {
        try {
            WifiManager wm = (WifiManager) getApplicationContext()
                    .getSystemService(Context.WIFI_SERVICE);
            if (wm == null) return "unknown";
            WifiInfo info = wm.getConnectionInfo();
            String ssid = info.getSSID();
            if (ssid == null || ssid.equals("<unknown ssid>")) return "unknown";
            if (ssid.startsWith("\"") && ssid.endsWith("\""))
                ssid = ssid.substring(1, ssid.length() - 1);
            return ssid;
        } catch (Exception e) {
            return "unknown";
        }
    }

    /**
     * Identify which floor to start on.
     *
     * Primary: restore the floor saved in SharedPreferences by the previous session (reliable,
     *          immune to barometer weather drift).
     * Fallback: if no saved floor exists (first launch), retry barometer-pressure matching
     *           up to 10 times at 1.5s intervals before defaulting to floor 0.
     */
    private void identifyStartingFloor() {
        if (currentBuilding == null) return;

        // --- Primary: restore last-known floor ---
        int savedFloor = getSharedPreferences(PREF_MAP_CONTEXT, Context.MODE_PRIVATE)
                .getInt(PREF_LAST_FLOOR, PREF_NO_FLOOR);
        if (savedFloor != PREF_NO_FLOOR && currentBuilding.hasFloor(savedFloor)) {
            Log.d("MapContext", "Restoring saved floor: " + savedFloor);
            switchToFloor(savedFloor);
            return;
        }

        // --- Fallback: first launch — try barometer pressure matching ---
        Log.d("MapContext", "No saved floor — attempting barometer identification");
        final int[] attemptsLeft = {10};
        Runnable[] retryRef = new Runnable[1];
        retryRef[0] = () -> {
            if (currentBuilding == null) return;
            if (barometerDetector != null) {
                float pressure = barometerDetector.getLastSmoothed();
                if (!Float.isNaN(pressure)) {
                    int floor = buildingMapRepo.identifyFloor(currentBuilding, pressure);
                    Log.d("MapContext", "Barometer identified floor " + floor
                            + " (pressure=" + pressure + " hPa, attempt "
                            + (11 - attemptsLeft[0]) + ")");
                    switchToFloor(floor);
                    return;
                }
            }
            attemptsLeft[0]--;
            if (attemptsLeft[0] > 0) {
                new Handler(Looper.getMainLooper()).postDelayed(retryRef[0], 1500);
            } else {
                Log.w("MapContext", "Barometer timed out — defaulting to floor 0");
                switchToFloor(0);
            }
        };
        new Handler(Looper.getMainLooper()).postDelayed(retryRef[0], 1500);
    }

    /**
     * Load the zone/alert repos for the given floor and reset positioning state.
     * All subsequent zone+alert operations use the newly scoped repositories.
     */
    private void switchToFloor(int floor) {
        if (currentBuilding == null) return;
        currentMapFloor = floor;
        String buildingId = currentBuilding.id;

        // Scope repos to this building + floor
        zoneRepo  = new ZoneMarkerRepository(this,
                BuildingMapRepository.zoneKey(buildingId, floor));
        alertRepo = new ZoneAlertRepository(this,
                BuildingMapRepository.alertKey(buildingId, floor));

        // Reset positioning: restore to exit zone if returning, otherwise start at origin
        if (locationMapView != null) locationMapView.clearPath();
        float[] returnPos = lastZonePerFloor.get(floor);
        if (returnPos != null && stepDetectionService != null) {
            Log.d("MapContext", String.format(
                    "Restoring floor %d position to recorded exit zone (%.2f, %.2f)",
                    floor, returnPos[0], returnPos[1]));
            stepDetectionService.resetToPosition(returnPos[0], returnPos[1]);
        } else if (stepDetectionService != null) {
            stepDetectionService.resetToOrigin();
        }

        // Reload UI
        currentZoneMarkerId = null;
        snoozedAlertIds.clear();
        refreshMapMarkers();
        updateServiceZonePositions();

        String floorLabel = BuildingMapRepository.floorLabel(floor);
        runOnUiThread(() -> {
            Toast.makeText(this,
                    "📍 " + currentBuilding.buildingName + " · " + floorLabel,
                    Toast.LENGTH_SHORT).show();
            if (tvCurrentZone != null) tvCurrentZone.setVisibility(View.GONE);
        });
        Log.d("MapContext", "Switched to " + buildingId + " / floor " + floor);
    }

    /**
     * Called by the barometer listener when a floor change is detected.
     * Records the current zone position so we can restore it when returning,
     * then registers/loads the new floor and switches to it.
     */
    private void handleBarometerFloorChange(int newFloor, String direction) {
        if (currentBuilding == null) return;

        // --- Record exit zone before leaving this floor ---
        if (zoneRepo != null && currentZoneMarkerId != null) {
            for (ZoneMarker m : zoneRepo.getAll()) {
                if (m.id.equals(currentZoneMarkerId)) {
                    lastZonePerFloor.put(currentMapFloor, new float[]{m.x, m.y});
                    Log.d("MapContext", String.format(
                            "Exit zone recorded for floor %d: '%s' at (%.2f, %.2f)",
                            currentMapFloor, m.name, m.x, m.y));
                    break;
                }
            }
        }

        boolean isNew = !currentBuilding.hasFloor(newFloor);
        if (isNew) {
            float pressure = (barometerDetector != null)
                    ? barometerDetector.getLastSmoothed() : 0f;
            buildingMapRepo.addFloor(currentBuilding.id, newFloor, pressure);
            currentBuilding = buildingMapRepo.findBySsid(currentBuilding.ssid);
            Log.d("MapContext", "New floor registered: " + newFloor
                    + " (" + pressure + " hPa)");
        }

        String label = BuildingMapRepository.floorLabel(newFloor);
        String msg = isNew
                ? "New floor created: " + label
                : "Floor " + direction + " → " + label;
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
        switchToFloor(newFloor);
    }

    private void showNewLocationDialog(String ssid) {
        EditText input = new EditText(this);
        input.setHint("Building / Location name");
        input.setText(ssid);

        new AlertDialog.Builder(this)
                .setTitle("📍 New Location Detected")
                .setMessage("WiFi: " + ssid + "\nName this building or location:")
                .setView(input)
                .setPositiveButton("Create Map", (d, w) -> {
                    String name = input.getText().toString().trim();
                    if (name.isEmpty()) name = ssid;
                    float p = (barometerDetector != null)
                            ? barometerDetector.getLastSmoothed() : 1013.25f;
                    currentBuilding = buildingMapRepo.createBuilding(ssid, name,
                            Float.isNaN(p) ? 1013.25f : p);
                    switchToFloor(0);
                })
                .setNeutralButton("Copy from Existing", (d, w) -> showDuplicateDialog(ssid))
                .setNegativeButton("Skip", (d, w) -> {
                    float p = (barometerDetector != null)
                            ? barometerDetector.getLastSmoothed() : 1013.25f;
                    currentBuilding = buildingMapRepo.createBuilding(ssid, ssid,
                            Float.isNaN(p) ? 1013.25f : p);
                    switchToFloor(0);
                })
                .show();
    }

    private void showDuplicateDialog(String ssid) {
        List<BuildingMap> buildings = buildingMapRepo.getAll();
        if (buildings.isEmpty()) { showNewLocationDialog(ssid); return; }

        String[] names = new String[buildings.size()];
        for (int i = 0; i < buildings.size(); i++) names[i] = buildings.get(i).buildingName;
        final int[] sel = {0};

        new AlertDialog.Builder(this)
                .setTitle("Copy zones from which building?")
                .setSingleChoiceItems(names, 0, (d, w) -> sel[0] = w)
                .setPositiveButton("Copy & Create", (d, w) -> {
                    BuildingMap src = buildings.get(sel[0]);
                    EditText nameInput = new EditText(this);
                    nameInput.setText(ssid);
                    new AlertDialog.Builder(this)
                            .setTitle("Name this building")
                            .setView(nameInput)
                            .setPositiveButton("Create", (d2, w2) -> {
                                String name = nameInput.getText().toString().trim();
                                if (name.isEmpty()) name = ssid;
                                float p = (barometerDetector != null)
                                        ? barometerDetector.getLastSmoothed() : 1013.25f;
                                currentBuilding = buildingMapRepo.createBuilding(ssid, name,
                                        Float.isNaN(p) ? 1013.25f : p);
                                buildingMapRepo.duplicateZones(
                                        src.id, 0, currentBuilding.id, 0);
                                switchToFloor(0);
                            })
                            .setNegativeButton("Cancel", null).show();
                })
                .setNegativeButton("Cancel", (d, w) -> showNewLocationDialog(ssid))
                .show();
    }

    private void initStepDetectionService() {
        // Initialize the service
        stepDetectionService = new StepDetectionService(this);

        // Create and store the listener for proper cleanup
        mainActivityListener = new StepDetectionService.DeviceStateListener() {
            @Override
            public void onDeviceStateChanged(boolean isStationary, float confidence) {
                runOnUiThread(() -> {
                    if (tvState != null) {
                        tvState.setText("State: " + (isStationary ? "STATIONARY" : "MOVING") +
                                " (Confidence: " + String.format("%.0f%%", confidence * 100) + ")");
                    }
                });
            }

            @Override
            public void onStepDetected(long timestamp, float stepLength,
                    int totalSteps, int sessionSteps) {
                runOnUiThread(() -> {
                    updateStepDisplay(totalSteps, sessionSteps, stepLength);
                });
            }

            @Override
            public void onStepsReset() {
                runOnUiThread(() -> {
                    if (tvSessionSteps != null) {
                        tvSessionSteps.setText("Session: 0");
                    }
                    Log("Session steps reset");
                });
            }

            @Override
            public void onInitialPositionEstimated(float x, float y, int floor) {
                Log(String.format("Initial position estimated: (%.1f, %.1f) Floor: %d", x, y, floor));
            }

            @Override
            public void onCalibrationNodeMatched(float x, float y, String nodeId) {
                Log(String.format("Calibration node %s matched at (%.1f, %.1f)", nodeId, x, y));
            }

            @Override
            public void onMagneticDataCollected(float[] magneticVector) {
                // Magnetic data collected — used internally by the service.
            }

            @Override
            public void onActivityRecognized(String activity, float confidence) {
                runOnUiThread(() -> {
                    if (tvState != null) {
                        tvState.setText(String.format("Activity: %s (%.0f%%)",
                                activity, confidence * 100));
                    }
                    updateMlDebugPanel(activity, confidence);
                    Log(String.format("ML Activity: %s (%.1f%%)", activity, confidence * 100));
                });
            }

            @Override
            public void onWiFiRSSIUpdate(java.util.List<com.example.wytv2.wifi.WiFiReading> readings, float variance) {
                updateWiFiPanel(readings, variance);
            }

            @Override
            public void onPositionUpdated(com.example.wytv2.localization.Position fusedPosition) {
                // Track latest PDR position for zone saving
                currentPdrX = fusedPosition.x;
                currentPdrY = fusedPosition.y;

                runOnUiThread(() -> {
                    float heading = 0.0f;
                    if (stepDetectionService != null) {
                        heading = stepDetectionService.getCurrentHeading();
                    }

                    // Update map view
                    if (locationMapView != null) {
                        locationMapView.updatePosition(fusedPosition.x, fusedPosition.y, heading);
                    }

                    // Update position display
                    if (tvPositionCompact != null) {
                        tvPositionCompact.setText(String.format("Position: (%.1f, %.1f)",
                                fusedPosition.x, fusedPosition.y));
                    }

                    // Update heading display
                    if (tvHeadingCompact != null) {
                        tvHeadingCompact.setText(String.format("Heading: %.0f°", heading));
                    }

                    // Update deviation display if available
                    if (tvCurrentDeviation != null) {
                        float deviation = locationMapView != null ? locationMapView.getCurrentDeviation() : -1;
                        if (deviation >= 0) {
                            tvCurrentDeviation.setText(String.format("Deviation: %.2fm", deviation));
                        } else {
                            tvCurrentDeviation.setText("Deviation: --");
                        }
                    }

                    Log(String.format("Position: (%.2f, %.2f) Floor: %d",
                            fusedPosition.x, fusedPosition.y, fusedPosition.floor));
                });
            }
        };

        // Add the listener (new multi-listener API)
        stepDetectionService.addDeviceStateListener(mainActivityListener);

        // Load magnetic database
        stepDetectionService.loadMagneticDatabase("TestBuilding");

        // Start the service
        stepDetectionService.start();

        Log("Step detection service started with magnetic localization");
    }

    private void updateThresholdDisplay(float threshold) {
        // No threshold display in current layout.
    }

    private void updateStepDisplay(int totalSteps, int sessionSteps, float stepLength) {
        // No step count display in current layout.
    }



    @Override
    protected void onResume() {
        super.onResume();
        if (stepDetectionService != null) {
            // Re-add listener (it's removed in onPause)
            if (mainActivityListener != null) {
                stepDetectionService.addDeviceStateListener(mainActivityListener);
            }
            stepDetectionService.start();
            Log.d("MainActivity", "onResume: service started, listener re-added");
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (stepDetectionService != null) {
            if (mainActivityListener != null) {
                stepDetectionService.removeDeviceStateListener(mainActivityListener);
            }
            stepDetectionService.stop();
        }
    }

    @Override
    protected void onStop() {
        super.onStop();
        // Persist the current floor so the next startup restores to the right floor.
        if (currentBuilding != null) {
            getSharedPreferences(PREF_MAP_CONTEXT, Context.MODE_PRIVATE)
                    .edit()
                    .putInt(PREF_LAST_FLOOR, currentMapFloor)
                    .apply();
            Log.d("MapContext", "Saved last floor: " + currentMapFloor);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (stepDetectionService != null) {
            stepDetectionService.stop();
        }
    }

    /**
     * Update ML debug panel with current model activity.
     */
    private void updateMlDebugPanel(String activity, float confidence) {
        // Update activity prediction with emoji
        if (tvMlActivityPrediction != null) {
            String emoji = getActivityEmoji(activity);
            tvMlActivityPrediction.setText(String.format("%s Activity: %s (%.0f%%)",
                    emoji, activity, confidence * 100));
        }

        // Update threshold status
        if (tvMlThresholdStatus != null && stepDetectionService != null) {
            float threshold = stepDetectionService.getBinaryThreshold();
            tvMlThresholdStatus.setText(String.format("⚙️ Threshold: %.2f m/s²", threshold));
        }

        // Update buffer status
        if (tvMlBufferStatus != null) {
            tvMlBufferStatus.setText("📊 Buffer: Ready (50/50)");
        }

        // Update last update time
        if (tvMlLastUpdate != null) {
            java.text.SimpleDateFormat sdf = new java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.US);
            String currentTime = sdf.format(new java.util.Date());
            tvMlLastUpdate.setText(String.format("🕒 Updated: %s", currentTime));
        }
    }

    /**
     * Get emoji for activity type.
     */
    private String getActivityEmoji(String activity) {
        switch (activity) {
            case "Walking":
                return "🚶";
            case "Running":
                return "🏃";
            case "Stationary":
                return "🧍";
            case "Stairs Up":
                return "⬆️";
            case "Stairs Down":
                return "⬇️";
            default:
                return "❓";
        }
    }

    /**
     * Update WiFi RSSI panel with current scan data.
     */
    private void updateWiFiPanel(java.util.List<com.example.wytv2.wifi.WiFiReading> readings, float variance) {
        if (readings == null || readings.isEmpty()) {
            if (tvWifiApCount != null) {
                tvWifiApCount.setText("Access Points: 0");
            }
            if (tvWifiStatus != null) {
                tvWifiStatus.setText("Status: No WiFi detected");
            }
            return;
        }

        // Update AP count
        if (tvWifiApCount != null) {
            tvWifiApCount.setText(String.format("Access Points: %d", readings.size()));
        }

        // Update variance with status indicator
        if (tvWifiVariance != null) {
            String varianceColor = variance < 5.0f ? "🟢" : "🔴";
            tvWifiVariance.setText(String.format("%s Variance: %.1f dBm²", varianceColor, variance));
        }

        // Update status based on variance
        if (tvWifiStatus != null) {
            if (variance < 5.0f) {
                tvWifiStatus.setText("Status: Stable (Stationary)");
            } else {
                tvWifiStatus.setText("Status: Changing (Moving)");
            }
        }

        // Find and display strongest AP
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
    }

    private void Log(String message) {
        android.util.Log.d("MainActivity", message);
    }

    // ---- Zone System --------------------------------------------------------

    private void showSaveZoneDialog() {
        android.widget.EditText input = new android.widget.EditText(this);
        input.setInputType(android.text.InputType.TYPE_CLASS_TEXT);
        input.setHint("e.g. Room A, Lab, Entrance");

        new android.app.AlertDialog.Builder(this)
                .setTitle("Save Zone")
                .setMessage(String.format("Current position: (%.1f, %.1f)", currentPdrX, currentPdrY))
                .setView(input)
                .setPositiveButton("Save", (dialog, which) -> {
                    String name = input.getText().toString().trim();
                    if (name.isEmpty()) name = "Zone " + (zoneRepo.getAll().size() + 1);
                    zoneRepo.save(name, currentPdrX, currentPdrY);
                    // Register the zone point as a SensorAnchor with current WiFi/magnetic snapshot
                    if (stepDetectionService != null) {
                        stepDetectionService.addZoneAnchor(currentPdrX, currentPdrY);
                    }
                    refreshMapMarkers();
                    updateServiceZonePositions();
                    android.widget.Toast.makeText(this, "Zone saved: " + name, android.widget.Toast.LENGTH_SHORT).show();
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void showManageZonesDialog() {
        java.util.List<com.example.wytv2.zones.ZoneMarker> markers = zoneRepo.getAll();
        if (markers.isEmpty()) {
            new android.app.AlertDialog.Builder(this)
                    .setTitle("Manage Zones")
                    .setMessage("No zones saved yet. Walk to a location and tap Save Zone.")
                    .setPositiveButton("OK", null)
                    .show();
            return;
        }

        String[] names = new String[markers.size()];
        for (int i = 0; i < markers.size(); i++) {
            com.example.wytv2.zones.ZoneMarker m = markers.get(i);
            names[i] = String.format("%s  (%.1f, %.1f)", m.name, m.x, m.y);
        }

        new android.app.AlertDialog.Builder(this)
                .setTitle("Manage Zones")
                .setItems(names, null)
                .setNeutralButton("Delete All", (d, w) -> new android.app.AlertDialog.Builder(this)
                        .setTitle("Delete all zones?")
                        .setPositiveButton("Delete", (d2, w2) -> {
                            zoneRepo.deleteAll();
                            refreshMapMarkers();
                            updateServiceZonePositions();
                            android.widget.Toast.makeText(this, "All zones deleted", android.widget.Toast.LENGTH_SHORT).show();
                        })
                        .setNegativeButton("Cancel", null)
                        .show())
                .setPositiveButton("Delete One", (d, w) -> showDeleteZoneChooser(markers))
                .setNegativeButton("Close", null)
                .show();
    }

    private void showDeleteZoneChooser(java.util.List<com.example.wytv2.zones.ZoneMarker> markers) {
        String[] names = new String[markers.size()];
        for (int i = 0; i < markers.size(); i++) {
            names[i] = String.format("%s  (%.1f, %.1f)", markers.get(i).name,
                    markers.get(i).x, markers.get(i).y);
        }
        new android.app.AlertDialog.Builder(this)
                .setTitle("Select zone to delete")
                .setItems(names, (d, which) -> {
                    String deleted = markers.get(which).name;
                    zoneRepo.delete(markers.get(which).id);
                    refreshMapMarkers();
                    updateServiceZonePositions();
                    android.widget.Toast.makeText(this, "Deleted: " + deleted, android.widget.Toast.LENGTH_SHORT).show();
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void refreshMapMarkers() {
        if (locationMapView != null) {
            locationMapView.setZoneMarkers(zoneRepo.getAll());
        }
    }

    /** Push current zone marker positions to the step detection service for zone-constrained correction. */
    private void updateServiceZonePositions() {
        if (stepDetectionService == null) return;
        java.util.List<com.example.wytv2.zones.ZoneMarker> markers = zoneRepo.getAll();
        java.util.List<float[]> positions = new java.util.ArrayList<>();
        for (com.example.wytv2.zones.ZoneMarker m : markers) positions.add(new float[]{m.x, m.y});
        stepDetectionService.setZonePositions(positions);
    }

    private void startZoneIdentification() {
        Runnable identifyZone = new Runnable() {
            @Override
            public void run() {
                if (zoneRepo != null) {
                    ZoneMarker nearest = zoneRepo.nearest(currentPdrX, currentPdrY, ZONE_RADIUS_M);
                    runOnUiThread(() -> {
                        String newZoneId = (nearest != null) ? nearest.id : null;

                        // Detect zone entry
                        if (newZoneId != null && !newZoneId.equals(currentZoneMarkerId)) {
                            // Entered a new zone — reset snooze tracking
                            snoozedAlertIds.clear();
                            // Check for active alerts
                            if (alertRepo != null && !alertDialogShowing) {
                                List<ZoneAlert> alerts = alertRepo.findActiveForZone(newZoneId);
                                for (ZoneAlert alert : alerts) {
                                    if (!snoozedAlertIds.contains(alert.id)) {
                                        showAlertFiredDialog(alert);
                                        break; // show one at a time
                                    }
                                }
                            }
                        } else if (newZoneId == null && currentZoneMarkerId != null) {
                            // Left any zone — clear snooze tracking for next entry
                            snoozedAlertIds.clear();
                        }

                        currentZoneMarkerId = newZoneId;

                        if (nearest != null) {
                            if (tvCurrentZone != null) {
                                tvCurrentZone.setText("📍 " + nearest.name);
                                tvCurrentZone.setVisibility(View.VISIBLE);
                            }
                            if (locationMapView != null) locationMapView.setActiveZone(nearest.id);
                        } else {
                            if (tvCurrentZone != null) tvCurrentZone.setVisibility(View.GONE);
                            if (locationMapView != null) locationMapView.setActiveZone(null);
                        }
                    });
                }
                zoneHandler.postDelayed(this, 1000);
            }
        };
        zoneHandler.postDelayed(identifyZone, 1000);
    }

    /** Called when the user has entered a zone that has an active alert. */
    private void showAlertFiredDialog(ZoneAlert alert) {
        alertDialogShowing = true;
        new android.app.AlertDialog.Builder(this)
                .setTitle("🔔 Zone Alert: " + alert.zoneName)
                .setMessage(alert.message)
                .setCancelable(false)
                .setPositiveButton("Snooze", (d, w) -> {
                    // Keep alert active but skip for this visit
                    snoozedAlertIds.add(alert.id);
                    alertDialogShowing = false;
                })
                .setNegativeButton("Dismiss", (d, w) -> {
                    // Move to Recent
                    if (alertRepo != null) alertRepo.dismiss(alert.id);
                    alertDialogShowing = false;
                    Toast.makeText(this, "Alert dismissed — see Recent", Toast.LENGTH_SHORT).show();
                })
                .setOnDismissListener(d -> alertDialogShowing = false)
                .show();
    }

    /** Top-level alerts dialog: Active alerts + Recent (dismissed). */
    private void showZoneAlertsDialog() {
        new android.app.AlertDialog.Builder(this)
                .setTitle("🔔 Zone Alerts")
                .setItems(new String[]{"Set New Alert…", "Active Alerts", "Recent (Dismissed)"}, (d, which) -> {
                    if (which == 0) showSetAlertDialog();
                    else if (which == 1) showActiveAlertsDialog();
                    else showRecentAlertsDialog();
                })
                .setNegativeButton("Close", null)
                .show();
    }

    private void showSetAlertDialog() {
        if (currentBuilding == null) {
            new android.app.AlertDialog.Builder(this)
                    .setTitle("No Building")
                    .setMessage("No building loaded.")
                    .setPositiveButton("OK", null).show();
            return;
        }

        // Collect zones from ALL floors of the current building
        // Each entry carries: display name, zone id, floor's alertRepo
        java.util.List<String> allZoneNames = new java.util.ArrayList<>();
        java.util.List<String> allZoneIds   = new java.util.ArrayList<>();
        java.util.List<com.example.wytv2.zones.ZoneAlertRepository> allAlertRepos =
                new java.util.ArrayList<>();

        java.util.List<FloorRecord> floors = new java.util.ArrayList<>(currentBuilding.floors);
        floors.sort(java.util.Comparator.comparingInt(f -> f.floorNumber));

        for (FloorRecord fr : floors) {
            String floorKey = BuildingMapRepository.alertKey(currentBuilding.id, fr.floorNumber);
            String zoneKey  = BuildingMapRepository.zoneKey(currentBuilding.id, fr.floorNumber);
            com.example.wytv2.zones.ZoneMarkerRepository floorZoneRepo =
                    new com.example.wytv2.zones.ZoneMarkerRepository(this, zoneKey);
            com.example.wytv2.zones.ZoneAlertRepository floorAlertRepo =
                    new com.example.wytv2.zones.ZoneAlertRepository(this, floorKey);
            String floorLabel = BuildingMapRepository.floorLabel(fr.floorNumber);
            boolean isCurrentFloor = (fr.floorNumber == currentMapFloor);
            for (com.example.wytv2.zones.ZoneMarker zm : floorZoneRepo.getAll()) {
                allZoneNames.add(isCurrentFloor
                        ? zm.name                                        // no prefix for current floor
                        : "[" + floorLabel + "] " + zm.name);
                allZoneIds.add(zm.id);
                allAlertRepos.add(floorAlertRepo);
                // store name for dialog label (used in the next step)
            }
        }

        if (allZoneNames.isEmpty()) {
            new android.app.AlertDialog.Builder(this)
                    .setTitle("No Zones")
                    .setMessage("Save at least one zone first before creating an alert.")
                    .setPositiveButton("OK", null).show();
            return;
        }

        String[] zoneNamesArr = allZoneNames.toArray(new String[0]);
        final int[] selected = {0};
        new android.app.AlertDialog.Builder(this)
                .setTitle("🔔 Select Zone")
                .setSingleChoiceItems(zoneNamesArr, 0, (d, which) -> selected[0] = which)
                .setPositiveButton("Next", (d, w) -> {
                    int idx = selected[0];
                    String zoneName = allZoneNames.get(idx);
                    String zoneId   = allZoneIds.get(idx);
                    com.example.wytv2.zones.ZoneAlertRepository targetAlertRepo =
                            allAlertRepos.get(idx);
                    // Message input
                    android.widget.EditText input = new android.widget.EditText(this);
                    input.setHint("Alert message…");
                    input.setInputType(android.text.InputType.TYPE_CLASS_TEXT);
                    new android.app.AlertDialog.Builder(this)
                            .setTitle("🔔 Alert for: " + zoneName)
                            .setView(input)
                            .setPositiveButton("Save Alert", (d2, w2) -> {
                                String msg = input.getText().toString().trim();
                                if (msg.isEmpty()) msg = "You have entered " + zoneName;
                                targetAlertRepo.saveAlert(zoneId, zoneName, msg);
                                // Reload the field-level alertRepo so its in-memory cache
                                // picks up the new alert (targetAlertRepo is a separate instance
                                // that shares the same SharedPreferences key for the current floor).
                                if (alertRepo != null) alertRepo.reload();
                                Toast.makeText(this, "Alert set for " + zoneName,
                                        Toast.LENGTH_SHORT).show();
                            })
                            .setNegativeButton("Cancel", null)
                            .show();
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void showActiveAlertsDialog() {
        List<ZoneAlert> aktiv = alertRepo.getActiveAlerts();
        if (aktiv.isEmpty()) {
            new android.app.AlertDialog.Builder(this)
                    .setTitle("Active Alerts")
                    .setMessage("No active alerts. Tap 'Set New Alert…' to add one.")
                    .setPositiveButton("OK", null).show();
            return;
        }
        String[] items = new String[aktiv.size()];
        for (int i = 0; i < aktiv.size(); i++)
            items[i] = "🔔 " + aktiv.get(i).zoneName + ": " + aktiv.get(i).message;

        new android.app.AlertDialog.Builder(this)
                .setTitle("Active Alerts")
                .setItems(items, null)
                .setPositiveButton("Delete One…", (d, w) -> {
                    new android.app.AlertDialog.Builder(this)
                            .setTitle("Select alert to delete")
                            .setItems(items, (d2, which) -> {
                                alertRepo.delete(aktiv.get(which).id);
                                Toast.makeText(this, "Alert deleted", Toast.LENGTH_SHORT).show();
                            })
                            .setNegativeButton("Cancel", null).show();
                })
                .setNegativeButton("Close", null).show();
    }

    private void showRecentAlertsDialog() {
        List<ZoneAlert> dismissed = alertRepo.getDismissedAlerts();
        if (dismissed.isEmpty()) {
            new android.app.AlertDialog.Builder(this)
                    .setTitle("Recent Alerts")
                    .setMessage("No recently dismissed alerts.")
                    .setPositiveButton("OK", null).show();
            return;
        }
        String[] items = new String[dismissed.size()];
        for (int i = 0; i < dismissed.size(); i++)
            items[i] = dismissed.get(i).zoneName + ": " + dismissed.get(i).message;

        new android.app.AlertDialog.Builder(this)
                .setTitle("Recent (Dismissed) Alerts")
                .setItems(items, null)
                .setPositiveButton("Re-activate One…", (d, w) -> {
                    new android.app.AlertDialog.Builder(this)
                            .setTitle("Re-activate alert")
                            .setItems(items, (d2, which) -> {
                                alertRepo.reactivate(dismissed.get(which).id);
                                Toast.makeText(this, "Alert re-activated", Toast.LENGTH_SHORT).show();
                            })
                            .setNegativeButton("Cancel", null).show();
                })
                .setNeutralButton("Delete One…", (d, w) -> {
                    new android.app.AlertDialog.Builder(this)
                            .setTitle("Permanently delete")
                            .setItems(items, (d2, which) -> {
                                alertRepo.delete(dismissed.get(which).id);
                                Toast.makeText(this, "Alert permanently deleted", Toast.LENGTH_SHORT).show();
                            })
                            .setNegativeButton("Cancel", null).show();
                })
                .setNegativeButton("Close", null).show();
    }
}