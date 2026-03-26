package com.example.wytv2.mapcontext;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a mapped building/location, identified by its WiFi SSID.
 * Each building has one or more {@link FloorRecord}s, each with its own
 * barometer reference pressure and scoped zone/alert storage.
 */
public class BuildingMap {
    public String id;            // sanitized SSID — used as SharedPreferences key prefix
    public String ssid;          // raw SSID as returned by WifiManager
    public String buildingName;  // user-defined display name
    public List<FloorRecord> floors = new ArrayList<>();

    /** Find a floor record by floor number, or null if not registered. */
    public FloorRecord getFloor(int floorNumber) {
        for (FloorRecord f : floors)
            if (f.floorNumber == floorNumber) return f;
        return null;
    }

    /** True if a record for this floor number exists. */
    public boolean hasFloor(int floorNumber) {
        return getFloor(floorNumber) != null;
    }
}
