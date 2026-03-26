package com.example.wytv2.mapcontext;

/**
 * One floor of a {@link BuildingMap}.
 * {@code referencePressureHpa} is used on startup to identify which floor the user
 * is currently on before they move (matched by nearest barometer reading).
 */
public class FloorRecord {
    public int   floorNumber;          // 0 = ground, 1 = first up, -1 = basement
    public String label;               // "Ground Floor", "Floor 1", "Basement 1" …
    public float referencePressureHpa; // barometer reading (hPa) recorded when this floor was created
    public long  createdAt;

    public FloorRecord(int floorNumber, String label,
                       float referencePressureHpa, long createdAt) {
        this.floorNumber = floorNumber;
        this.label = label;
        this.referencePressureHpa = referencePressureHpa;
        this.createdAt = createdAt;
    }
}
