package com.example.wytv2.mapcontext;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

/**
 * Persists {@link BuildingMap}s (keyed by sanitized SSID) in SharedPreferences.
 *
 * Storage layout:
 *   "building_maps"          → JSON list of buildings + floor records
 *   "zones_<id>_f<floor>"   → zone marker JSON for that building+floor
 *   "alerts_<id>_f<floor>"  → zone alert JSON for that building+floor
 */
public class BuildingMapRepository {

    private static final String PREFS_NAME    = "building_maps";
    private static final String KEY_BUILDINGS = "buildings";

    private final Context context;
    private final SharedPreferences prefs;
    private final List<BuildingMap> cache = new ArrayList<>();

    public BuildingMapRepository(Context context) {
        this.context = context.getApplicationContext();
        prefs = this.context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        load();
    }

    // ---- Key helpers --------------------------------------------------------

    /** Sanitize an SSID into a safe SharedPreferences key prefix. */
    public static String sanitizeId(String ssid) {
        return ssid.replaceAll("[^a-zA-Z0-9_-]", "_").toLowerCase();
    }

    /** SharedPreferences key for zone markers of a specific building+floor. */
    public static String zoneKey(String buildingId, int floor) {
        return "zones_" + buildingId + "_f" + floor;
    }

    /** SharedPreferences key for zone alerts of a specific building+floor. */
    public static String alertKey(String buildingId, int floor) {
        return "alerts_" + buildingId + "_f" + floor;
    }

    /** Human-readable label for a floor number. */
    public static String floorLabel(int floor) {
        if (floor == 0)   return "Ground Floor";
        if (floor < 0)    return "Basement " + Math.abs(floor);
        return "Floor " + floor;
    }

    // ---- Read ---------------------------------------------------------------

    public synchronized List<BuildingMap> getAll() {
        return new ArrayList<>(cache);
    }

    public synchronized BuildingMap findBySsid(String ssid) {
        String id = sanitizeId(ssid);
        for (BuildingMap b : cache)
            if (b.id.equals(id)) return b;
        return null;
    }

    // ---- Write --------------------------------------------------------------

    /**
     * Create a new building with floor 0 already registered at {@code initialPressure}.
     * Replaces any existing building with the same sanitized SSID.
     */
    public synchronized BuildingMap createBuilding(String ssid, String buildingName,
                                                   float initialPressure) {
        BuildingMap b = new BuildingMap();
        b.id = sanitizeId(ssid);
        b.ssid = ssid;
        b.buildingName = buildingName;
        b.floors.add(new FloorRecord(0, "Ground Floor", initialPressure,
                System.currentTimeMillis()));
        cache.removeIf(e -> e.id.equals(b.id));
        cache.add(b);
        persist();
        return b;
    }

    /**
     * Register a new floor in an existing building, recording {@code pressureHpa}
     * as its reference. Replaces any existing record for that floor number.
     */
    public synchronized void addFloor(String buildingId, int floorNumber, float pressureHpa) {
        for (BuildingMap b : cache) {
            if (b.id.equals(buildingId)) {
                b.floors.removeIf(f -> f.floorNumber == floorNumber);
                b.floors.add(new FloorRecord(floorNumber, floorLabel(floorNumber),
                        pressureHpa, System.currentTimeMillis()));
                persist();
                return;
            }
        }
    }

    /**
     * Given the current barometer pressure, return the floor number whose
     * reference pressure is closest. Falls back to 0 if no records present.
     */
    public synchronized int identifyFloor(BuildingMap building, float currentPressure) {
        if (building.floors.isEmpty()) return 0;
        int bestFloor = building.floors.get(0).floorNumber;
        float bestDiff = Float.MAX_VALUE;
        for (FloorRecord f : building.floors) {
            float diff = Math.abs(currentPressure - f.referencePressureHpa);
            if (diff < bestDiff) { bestDiff = diff; bestFloor = f.floorNumber; }
        }
        return bestFloor;
    }

    /**
     * Copy zone marker JSON from one building+floor to another.
     * Used when duplicating an existing building's layout.
     */
    public synchronized void duplicateZones(String srcId, int srcFloor,
                                            String dstId, int dstFloor) {
        String srcJson = context
                .getSharedPreferences(zoneKey(srcId, srcFloor), Context.MODE_PRIVATE)
                .getString("markers", "[]");
        context.getSharedPreferences(zoneKey(dstId, dstFloor), Context.MODE_PRIVATE)
               .edit().putString("markers", srcJson).apply();
    }

    // ---- JSON persistence ---------------------------------------------------

    private void load() {
        cache.clear();
        String json = prefs.getString(KEY_BUILDINGS, "[]");
        try {
            JSONArray arr = new JSONArray(json);
            for (int i = 0; i < arr.length(); i++) {
                JSONObject o = arr.getJSONObject(i);
                BuildingMap b = new BuildingMap();
                b.id = o.getString("id");
                b.ssid = o.getString("ssid");
                b.buildingName = o.getString("buildingName");
                JSONArray fa = o.getJSONArray("floors");
                for (int j = 0; j < fa.length(); j++) {
                    JSONObject fo = fa.getJSONObject(j);
                    b.floors.add(new FloorRecord(
                            fo.getInt("floorNumber"),
                            fo.getString("label"),
                            (float) fo.getDouble("referencePressure"),
                            fo.getLong("createdAt")));
                }
                cache.add(b);
            }
        } catch (Exception e) {
            cache.clear();
        }
    }

    private void persist() {
        try {
            JSONArray arr = new JSONArray();
            for (BuildingMap b : cache) {
                JSONObject o = new JSONObject();
                o.put("id", b.id);
                o.put("ssid", b.ssid);
                o.put("buildingName", b.buildingName);
                JSONArray fa = new JSONArray();
                for (FloorRecord f : b.floors) {
                    JSONObject fo = new JSONObject();
                    fo.put("floorNumber", f.floorNumber);
                    fo.put("label", f.label);
                    fo.put("referencePressure", f.referencePressureHpa);
                    fo.put("createdAt", f.createdAt);
                    fa.put(fo);
                }
                o.put("floors", fa);
                arr.put(o);
            }
            prefs.edit().putString(KEY_BUILDINGS, arr.toString()).apply();
        } catch (Exception ignored) {}
    }
}
