package com.example.wytv2.zones;

/**
 * A named zone marker placed at a PDR coordinate on the map.
 * Persisted as JSON via ZoneMarkerRepository.
 */
public class ZoneMarker {
    public String id;           // unique UUID
    public String name;         // user-chosen name ("Room A", "Lab", …)
    public float x;             // PDR x in metres
    public float y;             // PDR y in metres
    public float radius = 3.0f; // zone radius in metres (default 3m, user-adjustable via pinch)
    public long createdAt;      // epoch ms

    public ZoneMarker(String id, String name, float x, float y, long createdAt) {
        this.id = id;
        this.name = name;
        this.x = x;
        this.y = y;
        this.createdAt = createdAt;
    }
}
