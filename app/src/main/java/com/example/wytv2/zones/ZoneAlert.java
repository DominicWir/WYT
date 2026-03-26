package com.example.wytv2.zones;

/**
 * A named alert tied to a zone marker. When the user enters the zone,
 * the alert fires. It can be snoozed (skip this visit, fire again next time)
 * or dismissed (moved to "recent" history). Dismissed alerts can be
 * reactivated or permanently deleted.
 */
public class ZoneAlert {
    public String id;           // UUID
    public String zoneMarkerId; // ZoneMarker.id this alert is tied to
    public String zoneName;     // cached display name
    public String message;      // user-defined alert text
    public boolean active;      // true = will fire; false = recently dismissed
    public long createdAt;
    public long dismissedAt;    // 0 if never dismissed

    public ZoneAlert(String id, String zoneMarkerId, String zoneName,
                     String message, boolean active, long createdAt, long dismissedAt) {
        this.id = id;
        this.zoneMarkerId = zoneMarkerId;
        this.zoneName = zoneName;
        this.message = message;
        this.active = active;
        this.createdAt = createdAt;
        this.dismissedAt = dismissedAt;
    }
}
