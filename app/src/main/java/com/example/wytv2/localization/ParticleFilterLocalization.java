package com.example.wytv2.localization;

import android.util.Log;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Particle Filter for indoor localization.
 * Fuses PDR (motion model), WiFi, and Magnetic field measurements.
 * 
 * Algorithm:
 * 1. PREDICT: Move particles based on step detection (PDR)
 * 2. UPDATE: Weight particles based on WiFi/Magnetic measurements
 * 3. RESAMPLE: Keep high-weight particles, discard low-weight ones
 * 4. ESTIMATE: Return weighted mean position
 */
public class ParticleFilterLocalization {
    private static final String TAG = "ParticleFilter";

    // Particle filter parameters
    private static final int NUM_PARTICLES = 500;
    private static final float MOTION_NOISE_X = 0.05f; // meters (reduced from 0.1)
    private static final float MOTION_NOISE_Y = 0.05f; // meters (reduced from 0.1)
    private static final float MOTION_NOISE_HEADING = 0.03f; // radians (~1.7 degrees, reduced from 0.05)

    // Measurement noise (standard deviation)
    private static final float WIFI_MEASUREMENT_NOISE = 2.0f; // meters (reduced from 3.0)
    private static final float MAGNETIC_MEASUREMENT_NOISE = 2.5f; // meters (increased from 2.0)

    // Sensor fusion weights (increased WiFi influence)
    private static final float WIFI_WEIGHT = 0.75f; // increased from 0.6
    private static final float MAGNETIC_WEIGHT = 0.25f; // decreased from 0.4

    // Resampling threshold (effective sample size ratio)
    private static final float RESAMPLE_THRESHOLD = 0.3f; // Resample when Neff < 150 (lowered from 0.95)

    // State
    private List<Particle> particles;
    private int currentFloor;
    private Random random;
    private boolean isInitialized;

    public ParticleFilterLocalization() {
        particles = new ArrayList<>();
        random = new Random();
        isInitialized = false;
    }

    /**
     * Initialize particle filter with initial position estimate and heading.
     * Particles are scattered around the initial position with Gaussian noise.
     * 
     * @param initialPos     Initial position estimate
     * @param numParticles   Number of particles to create
     * @param initialHeading Initial heading in radians (0 = East, π/2 = North)
     */
    public void initialize(Position initialPos, int numParticles, float initialHeading) {
        particles.clear();
        currentFloor = initialPos.floor;

        // Create particles around initial position
        for (int i = 0; i < numParticles; i++) {
            float x = initialPos.x + gaussian(0, 1.0f); // ±1m initial uncertainty
            float y = initialPos.y + gaussian(0, 1.0f);

            // Initialize heading around current device heading (±10° uncertainty, was ±30°)
            float heading = initialHeading + gaussian(0, (float) Math.toRadians(10));

            // Normalize to [0, 2π]
            heading = (float) ((heading + 2 * Math.PI) % (2 * Math.PI));

            particles.add(new Particle(x, y, heading));
        }

        normalizeWeights();
        isInitialized = true;

        Log.d(TAG, String.format("Initialized with %d particles at (%.1f, %.1f) heading %.1f°",
                numParticles, initialPos.x, initialPos.y, Math.toDegrees(initialHeading)));
    }

    /**
     * Initialize particle filter with initial position (backward compatibility).
     * Uses random headings - only for testing purposes.
     */
    public void initialize(Position initialPos, int numParticles) {
        initialize(initialPos, numParticles, 0.0f);
    }

    /**
     * Initialize particle filter at a known anchor position with a custom spread.
     * Used for hard-snap anchor correction: tight spread (e.g. 0.3m) since the
     * anchor is a physically confirmed location.
     */
    public void initialize(Position initialPos, int numParticles, float initialHeading, float initialSpread) {
        particles.clear();
        currentFloor = initialPos.floor;
        for (int i = 0; i < numParticles; i++) {
            float x = initialPos.x + gaussian(0, initialSpread);
            float y = initialPos.y + gaussian(0, initialSpread);
            float heading = initialHeading + gaussian(0, (float) Math.toRadians(10));
            heading = (float) ((heading + 2 * Math.PI) % (2 * Math.PI));
            particles.add(new Particle(x, y, heading));
        }
        normalizeWeights();
        isInitialized = true;
        Log.d(TAG, String.format("Anchor snap: %d particles at (%.2f, %.2f) ±%.1fm heading %.1f°",
                numParticles, initialPos.x, initialPos.y, initialSpread, Math.toDegrees(initialHeading)));
    }

    /**
     * Prediction step: Move particles based on step detection.
     * 
     * @param stepLength Length of step in meters
     * @param heading    Heading change in radians
     */
    public void predict(float stepLength, float heading) {
        if (!isInitialized) {
            Log.w(TAG, "Predict called before initialization");
            return;
        }

        // Calculate particle spread before prediction
        float spreadBefore = calculateParticleSpread();

        for (Particle p : particles) {
            // Update heading
            p.heading += heading + gaussian(0, MOTION_NOISE_HEADING);

            // Normalize heading to [0, 2π]
            p.heading = (float) ((p.heading + 2 * Math.PI) % (2 * Math.PI));

            // Move particle forward in heading direction
            float dx = stepLength * (float) Math.cos(p.heading);
            float dy = stepLength * (float) Math.sin(p.heading);

            // Add motion noise
            p.x += dx + gaussian(0, MOTION_NOISE_X);
            p.y += dy + gaussian(0, MOTION_NOISE_Y);
        }

        // Log prediction details
        float spreadAfter = calculateParticleSpread();
        Log.d(TAG, String.format("Predict: step=%.2fm heading=%.1f° spread: %.2fm→%.2fm",
                stepLength, Math.toDegrees(heading), spreadBefore, spreadAfter));
    }

    /**
     * Update particle weights based on WiFi measurement.
     */
    public void updateWithWiFi(Position wifiEstimate, float wifiConfidence) {
        if (!isInitialized || wifiEstimate == null)
            return;

        // Log WiFi update details
        Position currentEst = getEstimatedPosition();
        float distanceToWiFi = (float) Math.sqrt(
                Math.pow(currentEst.x - wifiEstimate.x, 2) +
                        Math.pow(currentEst.y - wifiEstimate.y, 2));

        Log.d(TAG, String.format("WiFi Update: estimate=(%.1f,%.1f) conf=%.2f current=(%.1f,%.1f) dist=%.2fm",
                wifiEstimate.x, wifiEstimate.y, wifiConfidence,
                currentEst.x, currentEst.y, distanceToWiFi));

        updateWeights(wifiEstimate, WIFI_MEASUREMENT_NOISE, WIFI_WEIGHT * wifiConfidence);
    }

    /**
     * Update particle weights based on Magnetic field measurement.
     */
    public void updateWithMagnetic(Position magneticEstimate, float magneticConfidence) {
        if (!isInitialized || magneticEstimate == null)
            return;

        // Log Magnetic update details
        Position currentEst = getEstimatedPosition();
        float distanceToMagnetic = (float) Math.sqrt(
                Math.pow(currentEst.x - magneticEstimate.x, 2) +
                        Math.pow(currentEst.y - magneticEstimate.y, 2));

        Log.d(TAG, String.format("Magnetic Update: estimate=(%.1f,%.1f) conf=%.2f current=(%.1f,%.1f) dist=%.2fm",
                magneticEstimate.x, magneticEstimate.y, magneticConfidence,
                currentEst.x, currentEst.y, distanceToMagnetic));

        updateWeights(magneticEstimate, MAGNETIC_MEASUREMENT_NOISE,
                MAGNETIC_WEIGHT * magneticConfidence);
    }

    /**
     * Anchor-based position correction (ZUPT / loop closure).
     * Called when device becomes stationary near a previously visited stop.
     * Uses tight noise (1.0m) since anchors are derived from the device's own trajectory.
     *
     * @param anchorPos   Known anchor position to correct toward
     * @param searchRadius Distance from anchor to current estimate (for logging)
     */
    public void updateWithAnchor(Position anchorPos, float searchRadius) {
        if (!isInitialized || anchorPos == null) return;

        Position current = getEstimatedPosition();
        float dist = (float) Math.sqrt(
                Math.pow(current.x - anchorPos.x, 2) +
                Math.pow(current.y - anchorPos.y, 2));

        Log.d(TAG, String.format(
                "Anchor correction: current=(%.2f,%.2f) anchor=(%.2f,%.2f) dist=%.2fm",
                current.x, current.y, anchorPos.x, anchorPos.y, dist));

        // Tight noise = strong correction pull. Weight=1.0 for maximum influence.
        updateWeights(anchorPos, 1.0f, 1.0f);

        // Immediately resample to reinforce particles near the anchor
        resample();
    }

    /**
     * Update particle weights based on measurement.
     * Uses Gaussian likelihood function.
     */
    private void updateWeights(Position measurement, float noise, float weight) {
        float spreadBefore = calculateParticleSpread();
        float neffBefore = calculateEffectiveSampleSize();

        // Log measurement details
        Log.d(TAG, String.format("Measurement: (%.1f, %.1f) noise=%.1f weight=%.2f",
                measurement.x, measurement.y, noise, weight));

        // Track weight changes
        float minWeight = Float.MAX_VALUE;
        float maxWeight = 0;
        float minDistance = Float.MAX_VALUE;
        float maxDistance = 0;

        for (Particle p : particles) {
            float distance = p.distanceTo(measurement);
            minDistance = Math.min(minDistance, distance);
            maxDistance = Math.max(maxDistance, distance);

            // Gaussian likelihood: exp(-distance² / (2 * σ²))
            float likelihood = (float) Math.exp(
                    -(distance * distance) / (2 * noise * noise));

            // Update weight (multiplicative)
            p.weight *= Math.pow(likelihood, weight);

            minWeight = Math.min(minWeight, p.weight);
            maxWeight = Math.max(maxWeight, p.weight);
        }

        normalizeWeights();

        // Check if resampling needed
        float neff = calculateEffectiveSampleSize();
        float spreadAfter = calculateParticleSpread();
        float resampleThreshold = RESAMPLE_THRESHOLD * particles.size();

        // Enhanced logging
        Log.d(TAG, String.format("Update: spread %.2fm→%.2fm Neff %.0f→%.0f dist[%.1f-%.1f] weights[%.6f-%.6f]",
                spreadBefore, spreadAfter, neffBefore, neff, minDistance, maxDistance, minWeight, maxWeight));
        Log.d(TAG, String.format("Resample check: Neff=%.0f threshold=%.0f %s",
                neff, resampleThreshold, neff < resampleThreshold ? "→ RESAMPLING" : "(no resample)"));

        if (neff < resampleThreshold) {
            resample();
        }
    }

    /**
     * Resample particles using low variance resampling.
     * Keeps high-weight particles, discards low-weight ones.
     */
    private void resample() {
        List<Particle> newParticles = new ArrayList<>();
        int n = particles.size();

        // Low variance resampling
        float r = random.nextFloat() / n;
        float c = particles.get(0).weight;
        int i = 0;

        for (int m = 0; m < n; m++) {
            float u = r + m / (float) n;

            while (u > c && i < n - 1) {
                i++;
                c += particles.get(i).weight;
            }

            // Copy particle and add noise to prevent depletion
            Particle newP = particles.get(i).copy();
            newP.x += gaussian(0, 0.1f); // Reduced from 0.3f for tighter tracking
            newP.y += gaussian(0, 0.1f); // Reduced from 0.3f for tighter tracking
            newP.heading += gaussian(0, 0.02f); // Reduced from 0.05f
            newP.weight = 1.0f / n;

            newParticles.add(newP);
        }

        particles = newParticles;
        Log.d(TAG, "Resampled particles");
    }

    /**
     * Apply ML-based drift correction using relative displacement.
     *
     * Instead of using absolute ML positions (which are in a different coordinate space),
     * this method computes the displacement between consecutive ML predictions and
     * applies a fraction of it as a correction to all particles.
     *
     * @param deltaX  Relative displacement in X from ML model (meters-equivalent)
     * @param deltaY  Relative displacement in Y from ML model (meters-equivalent)
     * @param scale   Scale factor to convert ML displacement to local coordinates
     * @param alpha   Correction strength (0.0 = ignore, 1.0 = fully apply). Recommended: 0.3
     */
    public void applyMLDriftCorrection(float deltaX, float deltaY, float scale, float alpha) {
        if (!isInitialized) return;

        // Scale the ML displacement to local coordinate magnitude
        float correctionX = deltaX * scale * alpha;
        float correctionY = deltaY * scale * alpha;

        float magnitude = (float) Math.sqrt(correctionX * correctionX + correctionY * correctionY);

        // Skip tiny corrections (noise)
        if (magnitude < 0.01f) return;

        // Cap maximum single correction to prevent jumps
        float maxCorrection = 2.0f; // meters
        if (magnitude > maxCorrection) {
            float factor = maxCorrection / magnitude;
            correctionX *= factor;
            correctionY *= factor;
            magnitude = maxCorrection;
        }

        // Apply correction to all particles with noise
        for (Particle p : particles) {
            p.x += correctionX + gaussian(0, magnitude * 0.1f);
            p.y += correctionY + gaussian(0, magnitude * 0.1f);
        }

        Log.d(TAG, String.format(
                "ML Drift Correction: delta=(%.3f, %.3f) scaled=(%.3f, %.3f) magnitude=%.3fm",
                deltaX, deltaY, correctionX, correctionY, magnitude));
    }

    /**
     * Get current position estimate (weighted mean of particles).
     */
    public Position getEstimatedPosition() {
        if (!isInitialized || particles.isEmpty()) {
            return null;
        }

        float sumX = 0, sumY = 0, sumWeight = 0;
        float sumCosHeading = 0, sumSinHeading = 0;

        for (Particle p : particles) {
            sumX += p.x * p.weight;
            sumY += p.y * p.weight;
            sumCosHeading += Math.cos(p.heading) * p.weight;
            sumSinHeading += Math.sin(p.heading) * p.weight;
            sumWeight += p.weight;
        }

        float x = sumX / sumWeight;
        float y = sumY / sumWeight;

        // Confidence based on effective sample size
        float neff = calculateEffectiveSampleSize();
        float confidence = neff / particles.size();

        return new Position(x, y, currentFloor, confidence);
    }

    /**
     * Get current heading estimate (circular mean).
     */
    public float getEstimatedHeading() {
        if (!isInitialized || particles.isEmpty()) {
            return 0;
        }

        float sumCos = 0, sumSin = 0, sumWeight = 0;

        for (Particle p : particles) {
            sumCos += Math.cos(p.heading) * p.weight;
            sumSin += Math.sin(p.heading) * p.weight;
            sumWeight += p.weight;
        }

        return (float) Math.atan2(sumSin / sumWeight, sumCos / sumWeight);
    }

    /**
     * Calculate effective sample size (Neff).
     * Indicates particle diversity. Low Neff means particles are concentrated.
     */
    public float calculateEffectiveSampleSize() {
        float sumSquaredWeights = 0;

        for (Particle p : particles) {
            sumSquaredWeights += p.weight * p.weight;
        }

        return sumSquaredWeights > 0 ? 1.0f / sumSquaredWeights : 0;
    }

    /**
     * Normalize particle weights to sum to 1.
     */
    private void normalizeWeights() {
        float sumWeights = 0;

        for (Particle p : particles) {
            sumWeights += p.weight;
        }

        if (sumWeights > 0) {
            for (Particle p : particles) {
                p.weight /= sumWeights;
            }
        } else {
            // All weights are zero, reset to uniform
            float uniformWeight = 1.0f / particles.size();
            for (Particle p : particles) {
                p.weight = uniformWeight;
            }
        }
    }

    /**
     * Generate Gaussian random number with given mean and standard deviation.
     */
    private float gaussian(float mean, float stdDev) {
        return mean + (float) (random.nextGaussian() * stdDev);
    }

    /**
     * Check if particle filter is initialized.
     */
    public boolean isInitialized() {
        return isInitialized;
    }

    /**
     * Get number of particles.
     */
    public int getParticleCount() {
        return particles.size();
    }

    /**
     * Reset particle filter.
     */
    public void reset() {
        particles.clear();
        isInitialized = false;
        Log.d(TAG, "Particle filter reset");
    }

    /**
     * Calculate particle spread (standard deviation of particle positions).
     * Used for monitoring filter convergence.
     */
    public float calculateParticleSpread() {
        if (particles.isEmpty())
            return 0;

        // Calculate mean position
        float meanX = 0, meanY = 0;
        for (Particle p : particles) {
            meanX += p.x;
            meanY += p.y;
        }
        meanX /= particles.size();
        meanY /= particles.size();

        // Calculate variance
        float variance = 0;
        for (Particle p : particles) {
            float dx = p.x - meanX;
            float dy = p.y - meanY;
            variance += dx * dx + dy * dy;
        }
        variance /= particles.size();

        return (float) Math.sqrt(variance);
    }

    /**
     * Get detailed status for debugging.
     */
    public String getDebugStatus() {
        if (!isInitialized)
            return "Not initialized";

        Position pos = getEstimatedPosition();
        float spread = calculateParticleSpread();
        float neff = calculateEffectiveSampleSize();

        return String.format("Pos:(%.1f,%.1f) Spread:%.2fm Neff:%.0f/%.0f Conf:%.2f",
                pos.x, pos.y, spread, neff, (float) particles.size(), pos.confidence);
    }

    /**
     * Log particle distribution for debugging convergence issues.
     */
    public void logParticleDistribution() {
        if (!isInitialized || particles.isEmpty())
            return;

        // Find min/max positions
        float minX = Float.MAX_VALUE, maxX = -Float.MAX_VALUE;
        float minY = Float.MAX_VALUE, maxY = -Float.MAX_VALUE;

        for (Particle p : particles) {
            minX = Math.min(minX, p.x);
            maxX = Math.max(maxX, p.x);
            minY = Math.min(minY, p.y);
            maxY = Math.max(maxY, p.y);
        }

        Position est = getEstimatedPosition();
        float spread = calculateParticleSpread();

        Log.d(TAG, String.format("Particle Distribution: X[%.1f to %.1f] Y[%.1f to %.1f] Est:(%.1f,%.1f) Spread:%.2fm",
                minX, maxX, minY, maxY, est.x, est.y, spread));
    }
}
