package com.example.wytv2.pdr.graph;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.View;
import java.util.ArrayList;
import java.util.List;

public class GraphView extends View {
    // Data
    private List<Float> accelerationData = new ArrayList<>();
    private List<Long> timestampData = new ArrayList<>();
    private int maxDataPoints = 200; // Show last 200 points (~4 seconds at 50Hz)

    // Drawing properties
    private Paint linePaint;
    private Paint gridPaint;
    private Paint thresholdPaint;
    private Paint textPaint;
    private Paint pointPaint;

    // Graph properties
    private float graphMargin = 50;
    private float threshold = 1.2f;
    private boolean showThreshold = true;
    private boolean showPoints = false;

    // Colors
    private int backgroundColor = Color.BLACK;
    private int gridColor = Color.DKGRAY;
    private int lineColor = Color.GREEN;
    private int thresholdColor = Color.RED;
    private int textColor = Color.WHITE;
    private int pointColor = Color.YELLOW;

    public GraphView(Context context) {
        super(context);
        init();
    }

    public GraphView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        // Initialize paints
        linePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        linePaint.setColor(lineColor);
        linePaint.setStrokeWidth(3f);
        linePaint.setStyle(Paint.Style.STROKE);

        gridPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        gridPaint.setColor(gridColor);
        gridPaint.setStrokeWidth(1f);
        gridPaint.setStyle(Paint.Style.STROKE);

        thresholdPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        thresholdPaint.setColor(thresholdColor);
        thresholdPaint.setStrokeWidth(2f);
        thresholdPaint.setStyle(Paint.Style.STROKE);
        thresholdPaint.setAlpha(150);

        textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        textPaint.setColor(textColor);
        textPaint.setTextSize(24f);

        pointPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        pointPaint.setColor(pointColor);
        pointPaint.setStyle(Paint.Style.FILL);
    }

    public void addDataPoint(float acceleration, long timestamp) {
        accelerationData.add(acceleration);
        timestampData.add(timestamp);

        // Keep only last maxDataPoints
        if (accelerationData.size() > maxDataPoints) {
            accelerationData.remove(0);
            timestampData.remove(0);
        }

        // Request redraw
        invalidate();
    }

    public void setThreshold(float threshold) {
        this.threshold = threshold;
        invalidate();
    }

    public void setShowThreshold(boolean show) {
        this.showThreshold = show;
        invalidate();
    }

    public void setShowPoints(boolean show) {
        this.showPoints = show;
        invalidate();
    }

    public void clearData() {
        accelerationData.clear();
        timestampData.clear();
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        // Draw background
        canvas.drawColor(backgroundColor);

        if (accelerationData.isEmpty()) {
            drawNoDataMessage(canvas);
            return;
        }

        // Calculate graph bounds
        float graphWidth = getWidth() - 2 * graphMargin;
        float graphHeight = getHeight() - 2 * graphMargin;

        // Draw grid
        drawGrid(canvas, graphWidth, graphHeight);

        // Draw threshold line
        if (showThreshold) {
            drawThresholdLine(canvas, graphWidth, graphHeight);
        }

        // Draw acceleration line
        drawAccelerationLine(canvas, graphWidth, graphHeight);

        // Draw points if enabled
        if (showPoints) {
            drawDataPoints(canvas, graphWidth, graphHeight);
        }

        // Draw axis labels
        drawAxisLabels(canvas, graphWidth, graphHeight);

        // Draw stats
        drawStatistics(canvas);
    }

    private void drawNoDataMessage(Canvas canvas) {
        textPaint.setTextSize(32f);
        textPaint.setTextAlign(Paint.Align.CENTER);
        canvas.drawText("Waiting for acceleration data...",
                getWidth() / 2f, getHeight() / 2f, textPaint);
        textPaint.setTextSize(24f);
    }

    private void drawGrid(Canvas canvas, float graphWidth, float graphHeight) {
        // Vertical grid lines (time)
        int verticalLines = 10;
        for (int i = 0; i <= verticalLines; i++) {
            float x = graphMargin + (i * graphWidth / verticalLines);
            canvas.drawLine(x, graphMargin, x, graphMargin + graphHeight, gridPaint);
        }

        // Horizontal grid lines (acceleration)
        int horizontalLines = 8;
        for (int i = 0; i <= horizontalLines; i++) {
            float y = graphMargin + (i * graphHeight / horizontalLines);
            canvas.drawLine(graphMargin, y, graphMargin + graphWidth, y, gridPaint);
        }

        // Border
        canvas.drawRect(graphMargin, graphMargin,
                graphMargin + graphWidth, graphMargin + graphHeight, gridPaint);
    }

    private void drawThresholdLine(Canvas canvas, float graphWidth, float graphHeight) {
        float maxAcc = getMaxAcceleration();
        float minAcc = getMinAcceleration();
        float range = Math.max(maxAcc - minAcc, 1f);

        // Convert threshold to graph coordinates
        float normalizedThreshold = (threshold - minAcc) / range;
        float y = graphMargin + graphHeight * (1 - normalizedThreshold);

        // Draw threshold line
        canvas.drawLine(graphMargin, y, graphMargin + graphWidth, y, thresholdPaint);

        // Draw threshold label
        textPaint.setTextAlign(Paint.Align.LEFT);
        canvas.drawText(String.format("Threshold: %.2f", threshold),
                graphMargin + 10, y - 10, textPaint);
    }

    private void drawAccelerationLine(Canvas canvas, float graphWidth, float graphHeight) {
        if (accelerationData.size() < 2) return;

        float maxAcc = getMaxAcceleration();
        float minAcc = getMinAcceleration();
        float range = Math.max(maxAcc - minAcc, 1f);

        // Draw line connecting points
        float prevX = 0, prevY = 0;
        for (int i = 0; i < accelerationData.size(); i++) {
            // Normalize acceleration to 0-1 range
            float normalizedAcc = (accelerationData.get(i) - minAcc) / range;

            // Calculate coordinates
            float x = graphMargin + (i * graphWidth / (accelerationData.size() - 1));
            float y = graphMargin + graphHeight * (1 - normalizedAcc);

            // Draw line segment
            if (i > 0) {
                canvas.drawLine(prevX, prevY, x, y, linePaint);
            }

            prevX = x;
            prevY = y;
        }
    }

    private void drawDataPoints(Canvas canvas, float graphWidth, float graphHeight) {
        float maxAcc = getMaxAcceleration();
        float minAcc = getMinAcceleration();
        float range = Math.max(maxAcc - minAcc, 1f);

        for (int i = 0; i < accelerationData.size(); i++) {
            // Normalize acceleration
            float normalizedAcc = (accelerationData.get(i) - minAcc) / range;

            // Calculate coordinates
            float x = graphMargin + (i * graphWidth / (accelerationData.size() - 1));
            float y = graphMargin + graphHeight * (1 - normalizedAcc);

            // Draw point
            canvas.drawCircle(x, y, 4, pointPaint);

            // Draw value above threshold points
            if (accelerationData.get(i) > threshold) {
                canvas.drawText(String.format("%.2f", accelerationData.get(i)),
                        x, y - 10, textPaint);
            }
        }
    }

    private void drawAxisLabels(Canvas canvas, float graphWidth, float graphHeight) {
        // Y-axis label (acceleration)
        textPaint.setTextAlign(Paint.Align.RIGHT);
        float maxAcc = getMaxAcceleration();
        float minAcc = getMinAcceleration();

        canvas.drawText(String.format("%.1f", maxAcc),
                graphMargin - 10, graphMargin + 20, textPaint);
        canvas.drawText(String.format("%.1f", minAcc),
                graphMargin - 10, graphMargin + graphHeight - 5, textPaint);
        canvas.drawText("m/s²", graphMargin - 10, graphMargin + graphHeight/2, textPaint);

        // X-axis label (time)
        textPaint.setTextAlign(Paint.Align.CENTER);
        if (timestampData.size() > 1) {
            long timeSpan = timestampData.get(timestampData.size() - 1) - timestampData.get(0);
            canvas.drawText(String.format("%.1f s", timeSpan / 1000f),
                    graphMargin + graphWidth/2, graphMargin + graphHeight + 35, textPaint);
        }
    }

    private void drawStatistics(Canvas canvas) {
        if (accelerationData.isEmpty()) return;

        textPaint.setTextAlign(Paint.Align.LEFT);
        textPaint.setTextSize(20f);

        float yPos = graphMargin + 30;
        int lineHeight = 25;

        // Current acceleration
        float currentAcc = accelerationData.get(accelerationData.size() - 1);
        canvas.drawText(String.format("Current: %.3f m/s²", currentAcc),
                graphMargin + 10, yPos, textPaint);
        yPos += lineHeight;

        // Min/Max
        canvas.drawText(String.format("Min: %.3f, Max: %.3f",
                        getMinAcceleration(), getMaxAcceleration()),
                graphMargin + 10, yPos, textPaint);
        yPos += lineHeight;

        // Points above threshold
        int aboveThreshold = countPointsAboveThreshold();
        float percentage = (accelerationData.size() > 0) ?
                (aboveThreshold * 100f / accelerationData.size()) : 0;
        canvas.drawText(String.format("Above threshold: %d (%.1f%%)",
                        aboveThreshold, percentage),
                graphMargin + 10, yPos, textPaint);
        yPos += lineHeight;

        // Step frequency estimate
        if (timestampData.size() > 1) {
            long timeSpan = timestampData.get(timestampData.size() - 1) - timestampData.get(0);
            float frequency = (aboveThreshold > 1) ?
                    (aboveThreshold * 1000f / timeSpan) * 2 : 0;
            canvas.drawText(String.format("Step freq: %.1f Hz", frequency),
                    graphMargin + 10, yPos, textPaint);
        }
    }

    private float getMaxAcceleration() {
        if (accelerationData.isEmpty()) return 1f;
        float max = Float.MIN_VALUE;
        for (float acc : accelerationData) {
            if (acc > max) max = acc;
        }
        return Math.max(max, threshold * 1.5f); // Ensure threshold is visible
    }

    private float getMinAcceleration() {
        if (accelerationData.isEmpty()) return 0f;
        float min = Float.MAX_VALUE;
        for (float acc : accelerationData) {
            if (acc < min) min = acc;
        }
        return Math.min(min, 0f);
    }

    private int countPointsAboveThreshold() {
        int count = 0;
        for (float acc : accelerationData) {
            if (acc > threshold) count++;
        }
        return count;
    }
}