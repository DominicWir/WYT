package com.example.wytv2.localization;

/**
 * Represents a single particle in the particle filter.
 * Each particle is a hypothesis about the user's position and heading.
 */
public class Particle {
    public float x; // X position in meters
    public float y; // Y position in meters
    public float heading; // Heading in radians (0 = East, π/2 = North)
    public float weight; // Particle weight (importance)

    /**
     * Create a new particle at the given position and heading.
     */
    public Particle(float x, float y, float heading) {
        this.x = x;
        this.y = y;
        this.heading = heading;
        this.weight = 1.0f;
    }

    /**
     * Create a particle with specified weight.
     */
    public Particle(float x, float y, float heading, float weight) {
        this.x = x;
        this.y = y;
        this.heading = heading;
        this.weight = weight;
    }

    /**
     * Create a deep copy of this particle.
     */
    public Particle copy() {
        return new Particle(x, y, heading, weight);
    }

    /**
     * Calculate distance to a position.
     */
    public float distanceTo(float targetX, float targetY) {
        float dx = x - targetX;
        float dy = y - targetY;
        return (float) Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Calculate distance to a Position object.
     */
    public float distanceTo(Position pos) {
        return distanceTo(pos.x, pos.y);
    }

    @Override
    public String toString() {
        return String.format("Particle(%.2f, %.2f, θ=%.2f, w=%.4f)",
                x, y, heading, weight);
    }
}
