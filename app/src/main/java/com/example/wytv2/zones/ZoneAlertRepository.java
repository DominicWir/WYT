package com.example.wytv2.zones;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Persists ZoneAlerts as JSON in SharedPreferences.
 * Active alerts will fire on zone entry. Dismissed alerts appear in "Recent".
 */
public class ZoneAlertRepository {

    private static final String DEFAULT_PREFS_KEY = "zone_alerts";
    private static final String KEY_ALERTS = "alerts";

    private final SharedPreferences prefs;
    private final List<ZoneAlert> cache = new ArrayList<>();

    /** Backwards-compatible constructor. */
    public ZoneAlertRepository(Context context) {
        this(context, DEFAULT_PREFS_KEY);
    }

    /** Scoped constructor — use BuildingMapRepository.alertKey() to derive the key. */
    public ZoneAlertRepository(Context context, String prefsKey) {
        prefs = context.getApplicationContext()
                .getSharedPreferences(prefsKey, Context.MODE_PRIVATE);
        load();
    }

    // ---- Read ---------------------------------------------------------------

    public synchronized List<ZoneAlert> getActiveAlerts() {
        List<ZoneAlert> result = new ArrayList<>();
        for (ZoneAlert a : cache) if (a.active) result.add(a);
        return result;
    }

    public synchronized List<ZoneAlert> getDismissedAlerts() {
        List<ZoneAlert> result = new ArrayList<>();
        for (ZoneAlert a : cache) if (!a.active) result.add(a);
        result.sort((a, b) -> Long.compare(b.dismissedAt, a.dismissedAt));
        return result;
    }

    /** Returns active alerts for a specific zone marker. */
    public synchronized List<ZoneAlert> findActiveForZone(String zoneMarkerId) {
        List<ZoneAlert> result = new ArrayList<>();
        for (ZoneAlert a : cache)
            if (a.active && zoneMarkerId.equals(a.zoneMarkerId)) result.add(a);
        return result;
    }

    // ---- Write --------------------------------------------------------------

    public synchronized ZoneAlert saveAlert(String zoneMarkerId, String zoneName, String message) {
        ZoneAlert a = new ZoneAlert(UUID.randomUUID().toString(), zoneMarkerId, zoneName,
                message, true, System.currentTimeMillis(), 0);
        cache.add(a);
        persist();
        return a;
    }

    /** Move an alert to Recent (active=false). */
    public synchronized void dismiss(String alertId) {
        for (ZoneAlert a : cache) {
            if (a.id.equals(alertId)) {
                a.active = false;
                a.dismissedAt = System.currentTimeMillis();
                break;
            }
        }
        persist();
    }

    /** Restore a dismissed alert to active. */
    public synchronized void reactivate(String alertId) {
        for (ZoneAlert a : cache) {
            if (a.id.equals(alertId)) {
                a.active = true;
                a.dismissedAt = 0;
                break;
            }
        }
        persist();
    }

    /** Permanently delete an alert. */
    public synchronized void delete(String alertId) {
        cache.removeIf(a -> a.id.equals(alertId));
        persist();
    }

    // ---- JSON persistence ---------------------------------------------------

    private void load() {
        cache.clear();
        String json = prefs.getString(KEY_ALERTS, "[]");
        try {
            JSONArray arr = new JSONArray(json);
            for (int i = 0; i < arr.length(); i++) {
                JSONObject o = arr.getJSONObject(i);
                cache.add(new ZoneAlert(
                        o.getString("id"),
                        o.getString("zoneMarkerId"),
                        o.getString("zoneName"),
                        o.getString("message"),
                        o.getBoolean("active"),
                        o.getLong("createdAt"),
                        o.getLong("dismissedAt")));
            }
        } catch (Exception e) {
            cache.clear();
        }
    }

    private void persist() {
        try {
            JSONArray arr = new JSONArray();
            for (ZoneAlert a : cache) {
                JSONObject o = new JSONObject();
                o.put("id", a.id);
                o.put("zoneMarkerId", a.zoneMarkerId);
                o.put("zoneName", a.zoneName);
                o.put("message", a.message);
                o.put("active", a.active);
                o.put("createdAt", a.createdAt);
                o.put("dismissedAt", a.dismissedAt);
                arr.put(o);
            }
            prefs.edit().putString(KEY_ALERTS, arr.toString()).apply();
        } catch (Exception ignored) {}
    }
}
