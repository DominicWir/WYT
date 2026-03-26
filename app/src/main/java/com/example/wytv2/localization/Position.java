package com.example.wytv2.localization;

/**
 * Represents a 2D position with floor level and confidence score.
 * Used for indoor positioning with uncertainty estimation.
 */
public class Position {
    public float x; // X coordinate in meters
    public float y; // Y coordinate in meters
    public int floor; // Floor level (0 = ground floor)
    public float confidence; // Confidence score (0.0 to 1.0)
    public long timestamp; // Timestamp in milliseconds

    /**
     * Create a new position with confidence.
     */
    public Position(float x, float y, int floor, float confidence) {
        this.x = x;
        this.y = y;
        this.floor = floor;
        this.confidence = Math.max(0.0f, Math.min(1.0f, confidence)); // Clamp to [0,1]
        this.timestamp = System.currentTimeMillis();
    }

    /**
     * Create a position without confidence (defaults to 1.0).
     */
    public Position(float x, float y, int floor) {
        this(x, y, floor, 1.0f);
    }

    /**
     * Calculate Euclidean distance to another position.
     */
    public float distanceTo(Position other) {
        if (other == null)
            return Float.MAX_VALUE;

        float dx = x - other.x;
        float dy = y - other.y;
        return (float) Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Calculate distance to a point (x, y).
     */
    public float distanceTo(float otherX, float otherY) {
        float dx = x - otherX;
        float dy = y - otherY;
        return (float) Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Create a copy of this position.
     */
    public Position copy() {
        Position p = new Position(x, y, floor, confidence);
        p.timestamp = timestamp;
        return p;
    }

    @Override
    public String toString() {
        return String.format("Position(%.2f, %.2f, floor=%d, conf=%.2f)",
                x, y, floor, confidence);
    }
}
