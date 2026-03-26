package com.example.wytv2.visualization;

import java.util.ArrayList;
import java.util.List;

/**
 * Tracks position history for visualization.
 * Stores both estimated positions (from particle filter) and ground truth
 * positions (for testing).
 */
public class PathTracker {
    private static final int MAX_POINTS = 1000; // Limit to prevent memory issues

    private final List<PathPoint> estimatedPath;
    private final List<PathPoint> groundTruthPath;

    public static class PathPoint {
        public final float x;
        public final float y;
        public final float confidence; // 0.0 to 1.0
        public final long timestamp;

        public PathPoint(float x, float y, float confidence, long timestamp) {
            this.x = x;
            this.y = y;
            this.confidence = confidence;
            this.timestamp = timestamp;
        }
    }

    public PathTracker() {
        estimatedPath = new ArrayList<>();
        groundTruthPath = new ArrayList<>();
    }

    /**
     * Add an estimated position from the particle filter.
     */
    public void addEstimatedPosition(float x, float y, float confidence, long timestamp) {
        estimatedPath.add(new PathPoint(x, y, confidence, timestamp));

        // Remove oldest points if exceeding limit
        if (estimatedPath.size() > MAX_POINTS) {
            estimatedPath.remove(0);
        }
    }

    /**
     * Add a ground truth position (for testing/comparison).
     */
    public void addGroundTruthPosition(float x, float y, long timestamp) {
        groundTruthPath.add(new PathPoint(x, y, 1.0f, timestamp));

        // Remove oldest points if exceeding limit
        if (groundTruthPath.size() > MAX_POINTS) {
            groundTruthPath.remove(0);
        }
    }

    /**
     * Get the estimated path.
     */
    public List<PathPoint> getEstimatedPath() {
        return new ArrayList<>(estimatedPath);
    }

    /**
     * Get the ground truth path.
     */
    public List<PathPoint> getGroundTruthPath() {
        return new ArrayList<>(groundTruthPath);
    }

    /**
     * Get the current position (most recent estimated point).
     */
    public PathPoint getCurrentPosition() {
        if (estimatedPath.isEmpty()) {
            return null;
        }
        return estimatedPath.get(estimatedPath.size() - 1);
    }

    /**
     * Calculate current deviation from ground truth.
     * Returns -1 if no ground truth available.
     */
    public float getCurrentDeviation() {
        if (estimatedPath.isEmpty() || groundTruthPath.isEmpty()) {
            return -1;
        }

        // Get most recent points
        PathPoint estimated = estimatedPath.get(estimatedPath.size() - 1);
        PathPoint groundTruth = groundTruthPath.get(groundTruthPath.size() - 1);

        // Calculate Euclidean distance
        float dx = estimated.x - groundTruth.x;
        float dy = estimated.y - groundTruth.y;
        return (float) Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Get average deviation across all points.
     * Returns -1 if paths don't match in size.
     */
    public float getAverageDeviation() {
        if (estimatedPath.isEmpty() || groundTruthPath.isEmpty()) {
            return -1;
        }

        int minSize = Math.min(estimatedPath.size(), groundTruthPath.size());
        float totalDeviation = 0;

        for (int i = 0; i < minSize; i++) {
            PathPoint est = estimatedPath.get(i);
            PathPoint gt = groundTruthPath.get(i);

            float dx = est.x - gt.x;
            float dy = est.y - gt.y;
            totalDeviation += Math.sqrt(dx * dx + dy * dy);
        }

        return totalDeviation / minSize;
    }

    /**
     * Get bounds of all paths (for auto-scaling view).
     * Returns [minX, minY, maxX, maxY]
     */
    public float[] getBounds() {
        if (estimatedPath.isEmpty() && groundTruthPath.isEmpty()) {
            return new float[] { -1, -1, 1, 1 }; // Default bounds
        }

        float minX = Float.MAX_VALUE;
        float minY = Float.MAX_VALUE;
        float maxX = Float.MIN_VALUE;
        float maxY = Float.MIN_VALUE;

        // Check estimated path
        for (PathPoint p : estimatedPath) {
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x);
            maxY = Math.max(maxY, p.y);
        }

        // Check ground truth path
        for (PathPoint p : groundTruthPath) {
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x);
            maxY = Math.max(maxY, p.y);
        }

        // Add padding (10%)
        float paddingX = (maxX - minX) * 0.1f;
        float paddingY = (maxY - minY) * 0.1f;

        return new float[] {
                minX - paddingX,
                minY - paddingY,
                maxX + paddingX,
                maxY + paddingY
        };
    }

    /**
     * Clear all paths.
     */
    public void clear() {
        estimatedPath.clear();
        groundTruthPath.clear();
    }

    /**
     * Get number of estimated points.
     */
    public int getEstimatedPointCount() {
        return estimatedPath.size();
    }

    /**
     * Get number of ground truth points.
     */
    public int getGroundTruthPointCount() {
        return groundTruthPath.size();
    }
}
