package com.example.wytv2.zones;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Persists ZoneMarkers as a JSON array in SharedPreferences.
 * Thread-safe: all mutations are synchronised.
 */
public class ZoneMarkerRepository {

    private static final String DEFAULT_PREFS_KEY = "zone_markers";
    private static final String KEY_MARKERS = "markers";

    private final SharedPreferences prefs;
    private final List<ZoneMarker> cache = new ArrayList<>();

    /** Backwards-compatible constructor using the default prefs key. */
    public ZoneMarkerRepository(Context context) {
        this(context, DEFAULT_PREFS_KEY);
    }

    /** Scoped constructor — use BuildingMapRepository.zoneKey() to derive the key. */
    public ZoneMarkerRepository(Context context, String prefsKey) {
        prefs = context.getApplicationContext()
                .getSharedPreferences(prefsKey, Context.MODE_PRIVATE);
        load();
    }

    // ---- Read ---------------------------------------------------------------

    public synchronized List<ZoneMarker> getAll() {
        return new ArrayList<>(cache);
    }

    /**
     * Find the nearest marker within maxDistMetres.
     * Returns null if none found.
     */
    public synchronized ZoneMarker nearest(float x, float y, float maxDistMetres) {
        ZoneMarker best = null;
        float bestDist = maxDistMetres;
        for (ZoneMarker m : cache) {
            float dx = x - m.x;
            float dy = y - m.y;
            float d = (float) Math.sqrt(dx * dx + dy * dy);
            if (d < bestDist) {
                bestDist = d;
                best = m;
            }
        }
        return best;
    }

    // ---- Write --------------------------------------------------------------

    /** Add or update a marker. Returns the saved marker. */
    public synchronized ZoneMarker save(String name, float x, float y) {
        ZoneMarker m = new ZoneMarker(
                UUID.randomUUID().toString(), name, x, y, System.currentTimeMillis());
        cache.add(m);
        persist();
        return m;
    }

    public synchronized boolean delete(String id) {
        boolean removed = cache.removeIf(m -> m.id.equals(id));
        if (removed) persist();
        return removed;
    }

    /** Update an existing zone marker's name, x, and y in-place (for Move Zone). */
    public synchronized boolean update(ZoneMarker updated) {
        for (int i = 0; i < cache.size(); i++) {
            if (cache.get(i).id.equals(updated.id)) {
                cache.set(i, updated);
                persist();
                return true;
            }
        }
        return false;
    }

    public synchronized void deleteAll() {
        cache.clear();
        persist();
    }

    // ---- JSON persistence ---------------------------------------------------

    private void load() {
        cache.clear();
        String json = prefs.getString(KEY_MARKERS, "[]");
        try {
            JSONArray arr = new JSONArray(json);
            for (int i = 0; i < arr.length(); i++) {
                JSONObject o = arr.getJSONObject(i);
                cache.add(new ZoneMarker(
                        o.getString("id"),
                        o.getString("name"),
                        (float) o.getDouble("x"),
                        (float) o.getDouble("y"),
                        o.getLong("createdAt")));
                ZoneMarker zm = cache.get(cache.size() - 1);
                zm.radius = (float) o.optDouble("radius", 3.0);
            }
        } catch (Exception e) {
            cache.clear(); // corrupt data — start fresh
        }
    }

    private void persist() {
        try {
            JSONArray arr = new JSONArray();
            for (ZoneMarker m : cache) {
                JSONObject o = new JSONObject();
                o.put("id", m.id);
                o.put("name", m.name);
                o.put("x", m.x);
                o.put("y", m.y);
                o.put("radius", m.radius);
                o.put("createdAt", m.createdAt);
                arr.put(o);
            }
            prefs.edit().putString(KEY_MARKERS, arr.toString()).apply();
        } catch (Exception ignored) {}
    }
}
