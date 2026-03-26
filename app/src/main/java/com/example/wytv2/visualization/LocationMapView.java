package com.example.wytv2.visualization;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.DashPathEffect;
import android.util.AttributeSet;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;

import com.example.wytv2.zones.ZoneMarker;

import java.util.ArrayList;
import java.util.List;

/**
 * Custom view for displaying location tracking map with grid overlay.
 * Shows ground truth path, estimated path, and deviation indicators.
 */
public class LocationMapView extends View {

    private PathTracker pathTracker;

    // Zone markers
    private List<ZoneMarker> zoneMarkers = new ArrayList<>();
    private String activeZoneId = null;

    // Path visibility toggle
    private boolean showPath = true;

    // View transformation
    private float offsetX = 0;
    private float offsetY = 0;
    private float scale = 100; // pixels per metre — increased for better zone separation

    // Paints
    private Paint gridPaint;
    private Paint gridMajorPaint;
    private Paint axisPaint;
    private Paint groundTruthPaint;
    private Paint estimatedPaint;
    private Paint deviationPaint;
    private Paint currentPosPaint;
    private Paint textPaint;
    private Paint originPaint;
    private Paint zoneMarkerPaint;
    private Paint zoneActiveMarkerPaint;
    private Paint zoneTextPaint;
    private Paint zoneRadiusPaint;     // low-opacity fill for zone radius circle

    // Zone drag-edit mode
    /** Listener called when user finishes dragging a zone to a new position. */
    public interface ZoneDragListener {
        void onZoneMoved(ZoneMarker zone, float newX, float newY);
    }
    private ZoneMarker draggableZone = null;
    private ZoneDragListener zoneDragListener = null;
    private boolean isDraggingZone = false;
    /** Dedicated scale detector for pinch-to-resize the draggable zone radius (not the map). */
    private ScaleGestureDetector zoneScaleDetector;

    /** Highlight this zone as draggable. Set null to exit edit mode. */
    public void setDraggableZone(ZoneMarker zone, ZoneDragListener listener) {
        draggableZone = zone;
        zoneDragListener = listener;
        isDraggingZone = false;
        // Create a fresh zone-scale detector each time so no stale state leaks
        if (zone != null) {
            zoneScaleDetector = new ScaleGestureDetector(getContext(),
                    new ScaleGestureDetector.SimpleOnScaleGestureListener() {
                        @Override
                        public boolean onScale(ScaleGestureDetector d) {
                            if (draggableZone != null) {
                                draggableZone.radius *= d.getScaleFactor();
                                draggableZone.radius = Math.max(0.5f,
                                        Math.min(draggableZone.radius, 15f));
                                invalidate();
                            }
                            return true;
                        }
                    });
        } else {
            zoneScaleDetector = null;
        }
        invalidate();
    }

    // Gesture detectors
    private ScaleGestureDetector scaleDetector;
    private GestureDetector gestureDetector;

    // Auto-center flag
    private boolean autoCenter = true;

    public LocationMapView(Context context) {
        super(context);
        init();
    }

    public LocationMapView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        // Center the coordinate origin when the view is first sized
        if (oldw == 0 && oldh == 0) {
            offsetX = w / 2f;
            offsetY = h / 2f;
        }
    }

    private void init() {
        // Grid paint (minor lines)
        gridPaint = new Paint();
        gridPaint.setColor(0xFF333333);
        gridPaint.setStrokeWidth(1);
        gridPaint.setStyle(Paint.Style.STROKE);
        gridPaint.setAntiAlias(true);

        // Grid major paint (every 5m)
        gridMajorPaint = new Paint();
        gridMajorPaint.setColor(0xFF555555);
        gridMajorPaint.setStrokeWidth(2);
        gridMajorPaint.setStyle(Paint.Style.STROKE);
        gridMajorPaint.setAntiAlias(true);

        // Axis paint
        axisPaint = new Paint();
        axisPaint.setColor(0xFF888888);
        axisPaint.setStrokeWidth(3);
        axisPaint.setStyle(Paint.Style.STROKE);
        axisPaint.setAntiAlias(true);

        // Ground truth path (green)
        groundTruthPaint = new Paint();
        groundTruthPaint.setColor(0xFF4CAF50);
        groundTruthPaint.setStrokeWidth(4);
        groundTruthPaint.setStyle(Paint.Style.STROKE);
        groundTruthPaint.setAntiAlias(true);
        groundTruthPaint.setStrokeCap(Paint.Cap.ROUND);
        groundTruthPaint.setStrokeJoin(Paint.Join.ROUND);

        // Estimated path (blue)
        estimatedPaint = new Paint();
        estimatedPaint.setColor(0xFF2196F3);
        estimatedPaint.setStrokeWidth(4);
        estimatedPaint.setStyle(Paint.Style.STROKE);
        estimatedPaint.setAntiAlias(true);
        estimatedPaint.setStrokeCap(Paint.Cap.ROUND);
        estimatedPaint.setStrokeJoin(Paint.Join.ROUND);

        // Deviation lines (red, dashed)
        deviationPaint = new Paint();
        deviationPaint.setColor(0xFFFF5722);
        deviationPaint.setStrokeWidth(2);
        deviationPaint.setStyle(Paint.Style.STROKE);
        deviationPaint.setAntiAlias(true);
        deviationPaint.setPathEffect(new DashPathEffect(new float[] { 10, 10 }, 0));

        // Current position marker (yellow)
        currentPosPaint = new Paint();
        currentPosPaint.setColor(0xFFFFC107);
        currentPosPaint.setStyle(Paint.Style.FILL);
        currentPosPaint.setAntiAlias(true);

        // Text paint
        textPaint = new Paint();
        textPaint.setColor(0xFFCCCCCC);
        textPaint.setTextSize(24);
        textPaint.setAntiAlias(true);

        // Origin marker
        originPaint = new Paint();
        originPaint.setColor(0xFFFFFFFF);
        originPaint.setStyle(Paint.Style.FILL);
        originPaint.setAntiAlias(true);

        // Zone marker paint (magenta pin)
        zoneMarkerPaint = new Paint();
        zoneMarkerPaint.setColor(0xFFE040FB);
        zoneMarkerPaint.setStyle(Paint.Style.FILL);
        zoneMarkerPaint.setAntiAlias(true);

        // Active zone marker (bright yellow, filled)
        zoneActiveMarkerPaint = new Paint();
        zoneActiveMarkerPaint.setColor(0xFFFFEB3B);
        zoneActiveMarkerPaint.setStyle(Paint.Style.FILL);
        zoneActiveMarkerPaint.setAntiAlias(true);

        // Zone label text
        zoneTextPaint = new Paint();
        zoneTextPaint.setColor(0xFFFFFFFF);
        zoneTextPaint.setTextSize(28);
        zoneTextPaint.setAntiAlias(true);
        zoneTextPaint.setFakeBoldText(true);

        // Zone radius fill (low opacity)
        zoneRadiusPaint = new Paint();
        zoneRadiusPaint.setColor(0x22E040FB); // 13% opacity magenta
        zoneRadiusPaint.setStyle(Paint.Style.FILL);
        zoneRadiusPaint.setAntiAlias(true);

        // Setup gesture detectors
        scaleDetector = new ScaleGestureDetector(getContext(), new ScaleListener());
        gestureDetector = new GestureDetector(getContext(), new GestureListener());

        // Initialize path tracker
        pathTracker = new PathTracker();
    }

    public void setPathTracker(PathTracker tracker) {
        this.pathTracker = tracker;
        invalidate();
    }

    /** Update the displayed zone markers from the repository. */
    public void setZoneMarkers(List<ZoneMarker> markers) {
        zoneMarkers = new ArrayList<>(markers);
        invalidate();
    }

    /** Highlight the zone the user is currently inside. Pass null to clear. */
    public void setActiveZone(String zoneId) {
        activeZoneId = zoneId;
        invalidate();
    }

    /** Show or hide the estimated path (blue line). */
    public void setPathVisible(boolean visible) {
        showPath = visible;
        invalidate();
    }

    public boolean isPathVisible() { return showPath; }

    public void setAutoCenter(boolean autoCenter) {
        this.autoCenter = autoCenter;
        invalidate();
    }

    /**
     * Update the current position on the map.
     */
    public void updatePosition(float x, float y, float heading) {
        if (pathTracker != null) {
            // Use 1.0 confidence for now, or pass it in if available
            pathTracker.addEstimatedPosition(x, y, 1.0f, System.currentTimeMillis());

            if (autoCenter) {
                // Keep centered
                offsetX = getWidth() / 2 - x * scale;
                offsetY = getHeight() / 2 + y * scale;
            }
            invalidate();
        }
    }

    public void centerView() {
        if (pathTracker == null)
            return;

        float[] bounds = pathTracker.getBounds();
        float centerX = (bounds[0] + bounds[2]) / 2;
        float centerY = (bounds[1] + bounds[3]) / 2;

        offsetX = getWidth() / 2 - centerX * scale;
        offsetY = getHeight() / 2 + centerY * scale; // Y is inverted

        invalidate();
    }

    /**
     * Clear all paths.
     */
    public void clearPath() {
        if (pathTracker != null) {
            pathTracker.clear();
            invalidate();
        }
    }

    /**
     * Set a trigger point/landmark at the current position.
     * Currently implemented as adding a ground truth point for testing deviation.
     */
    public void setTriggerPoint() {
        if (pathTracker != null) {
            PathTracker.PathPoint current = pathTracker.getCurrentPosition();
            if (current != null) {
                pathTracker.addGroundTruthPosition(current.x, current.y, System.currentTimeMillis());
                invalidate();
            } else {
                // If no position yet, set at origin
                pathTracker.addGroundTruthPosition(0, 0, System.currentTimeMillis());
                invalidate();
            }
        }
    }

    /**
     * Get the current deviation from ground truth.
     */
    public float getCurrentDeviation() {
        return pathTracker != null ? pathTracker.getCurrentDeviation() : -1;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        // Black background
        canvas.drawColor(0xFF000000);

        // Auto-center on current position if enabled
        if (autoCenter && pathTracker != null) {
            PathTracker.PathPoint current = pathTracker.getCurrentPosition();
            if (current != null) {
                offsetX = getWidth() / 2 - current.x * scale;
                offsetY = getHeight() / 2 + current.y * scale;
            }
        }

        // Draw components
        drawGrid(canvas);
        drawAxes(canvas);
        drawOrigin(canvas);

        if (pathTracker != null) {
            drawGroundTruthPath(canvas);
            if (showPath) drawEstimatedPath(canvas);
            if (showPath) drawDeviationLines(canvas);
            drawCurrentPosition(canvas);
        }

        drawZoneMarkers(canvas);
        drawLegend(canvas);
    }

    private void drawGrid(Canvas canvas) {
        int width = getWidth();
        int height = getHeight();

        // Calculate visible range in meters
        float minX = -offsetX / scale;
        float maxX = (width - offsetX) / scale;
        float minY = -(height - offsetY) / scale;
        float maxY = -offsetY / scale;

        // Draw vertical grid lines (every 1m)
        for (int x = (int) Math.floor(minX); x <= Math.ceil(maxX); x++) {
            float screenX = offsetX + x * scale;
            Paint paint = (x % 5 == 0) ? gridMajorPaint : gridPaint;
            canvas.drawLine(screenX, 0, screenX, height, paint);
        }

        // Draw horizontal grid lines (every 1m)
        for (int y = (int) Math.floor(minY); y <= Math.ceil(maxY); y++) {
            float screenY = offsetY - y * scale;
            Paint paint = (y % 5 == 0) ? gridMajorPaint : gridPaint;
            canvas.drawLine(0, screenY, width, screenY, paint);
        }

        // Draw grid labels (every 5m)
        for (int x = (int) Math.floor(minX); x <= Math.ceil(maxX); x += 5) {
            if (x == 0)
                continue;
            float screenX = offsetX + x * scale;
            canvas.drawText(x + "m", screenX + 5, offsetY - 5, textPaint);
        }

        for (int y = (int) Math.floor(minY); y <= Math.ceil(maxY); y += 5) {
            if (y == 0)
                continue;
            float screenY = offsetY - y * scale;
            canvas.drawText(y + "m", offsetX + 5, screenY - 5, textPaint);
        }
    }

    private void drawAxes(Canvas canvas) {
        int width = getWidth();
        int height = getHeight();

        // X-axis
        canvas.drawLine(0, offsetY, width, offsetY, axisPaint);

        // Y-axis
        canvas.drawLine(offsetX, 0, offsetX, height, axisPaint);
    }

    private void drawOrigin(Canvas canvas) {
        // Draw origin marker (0, 0)
        float screenX = offsetX;
        float screenY = offsetY;
        canvas.drawCircle(screenX, screenY, 8, originPaint);
        canvas.drawText("(0,0)", screenX + 12, screenY - 12, textPaint);
    }

    private void drawGroundTruthPath(Canvas canvas) {
        List<PathTracker.PathPoint> path = pathTracker.getGroundTruthPath();
        if (path.size() < 2)
            return;

        Path pathObj = new Path();
        boolean first = true;

        for (PathTracker.PathPoint point : path) {
            float screenX = offsetX + point.x * scale;
            float screenY = offsetY - point.y * scale;

            if (first) {
                pathObj.moveTo(screenX, screenY);
                first = false;
            } else {
                pathObj.lineTo(screenX, screenY);
            }

            // Draw marker at each point
            canvas.drawCircle(screenX, screenY, 6, groundTruthPaint);
        }

        canvas.drawPath(pathObj, groundTruthPaint);
    }

    private void drawEstimatedPath(Canvas canvas) {
        List<PathTracker.PathPoint> path = pathTracker.getEstimatedPath();
        if (path.size() < 2)
            return;

        Path pathObj = new Path();
        boolean first = true;

        for (PathTracker.PathPoint point : path) {
            float screenX = offsetX + point.x * scale;
            float screenY = offsetY - point.y * scale;

            if (first) {
                pathObj.moveTo(screenX, screenY);
                first = false;
            } else {
                pathObj.lineTo(screenX, screenY);
            }

            // Draw marker with confidence-based alpha
            Paint markerPaint = new Paint(estimatedPaint);
            markerPaint.setStyle(Paint.Style.FILL);
            int alpha = (int) (point.confidence * 255);
            markerPaint.setAlpha(alpha);
            canvas.drawCircle(screenX, screenY, 6, markerPaint);
        }

        canvas.drawPath(pathObj, estimatedPaint);
    }

    private void drawDeviationLines(Canvas canvas) {
        List<PathTracker.PathPoint> estimated = pathTracker.getEstimatedPath();
        List<PathTracker.PathPoint> groundTruth = pathTracker.getGroundTruthPath();

        int minSize = Math.min(estimated.size(), groundTruth.size());

        for (int i = 0; i < minSize; i++) {
            PathTracker.PathPoint est = estimated.get(i);
            PathTracker.PathPoint gt = groundTruth.get(i);

            // Calculate deviation
            float dx = est.x - gt.x;
            float dy = est.y - gt.y;
            float deviation = (float) Math.sqrt(dx * dx + dy * dy);

            // Only draw if deviation > 0.5m
            if (deviation > 0.5f) {
                float estScreenX = offsetX + est.x * scale;
                float estScreenY = offsetY - est.y * scale;
                float gtScreenX = offsetX + gt.x * scale;
                float gtScreenY = offsetY - gt.y * scale;

                canvas.drawLine(gtScreenX, gtScreenY, estScreenX, estScreenY, deviationPaint);
            }
        }
    }

    private void drawCurrentPosition(Canvas canvas) {
        PathTracker.PathPoint current = pathTracker.getCurrentPosition();
        if (current == null)
            return;

        float screenX = offsetX + current.x * scale;
        float screenY = offsetY - current.y * scale;

        // Pulsing effect (simple version - just larger circle)
        canvas.drawCircle(screenX, screenY, 12, currentPosPaint);

        // Inner circle
        Paint innerPaint = new Paint(currentPosPaint);
        innerPaint.setColor(0xFF000000);
        canvas.drawCircle(screenX, screenY, 6, innerPaint);
    }

    /** Draw zone markers as labelled pins with radius circles. */
    private void drawZoneMarkers(Canvas canvas) {
        for (ZoneMarker marker : zoneMarkers) {
            float sx = offsetX + marker.x * scale;
            float sy = offsetY - marker.y * scale;
            boolean isActive  = marker.id.equals(activeZoneId);
            boolean isDragging = draggableZone != null && marker.id.equals(draggableZone.id);

            // Radius circle (drawn first, behind the pin)
            float radiusPx = marker.radius * scale;
            Paint rPaint = new Paint(zoneRadiusPaint);
            if (isDragging) rPaint.setColor(0x44FFEB3B); // golden tint when selected for drag
            canvas.drawCircle(sx, sy, radiusPx, rPaint);

            // Outer ring stroke
            Paint pinPaint = isDragging ? zoneActiveMarkerPaint
                             : isActive ? zoneActiveMarkerPaint : zoneMarkerPaint;
            Paint ring = new Paint(pinPaint);
            ring.setStyle(Paint.Style.STROKE);
            ring.setStrokeWidth(isDragging ? 5 : 3);
            canvas.drawCircle(sx, sy, 16, ring);

            // Filled dot
            canvas.drawCircle(sx, sy, 10, pinPaint);

            // Label
            float labelX = sx - zoneTextPaint.measureText(marker.name) / 2f;
            float labelY = sy - 22;
            Paint shadow = new Paint(zoneTextPaint);
            shadow.setColor(0xFF000000);
            canvas.drawText(marker.name, labelX + 1, labelY + 1, shadow);
            canvas.drawText(marker.name, labelX, labelY, zoneTextPaint);
        }
    }

    private void drawLegend(Canvas canvas) {
        int y = 30;
        int x = 10;

        // Ground truth
        canvas.drawLine(x, y, x + 30, y, groundTruthPaint);
        canvas.drawText("Ground Truth", x + 40, y + 5, textPaint);

        // Estimated
        y += 30;
        canvas.drawLine(x, y, x + 30, y, estimatedPaint);
        canvas.drawText("Estimated", x + 40, y + 5, textPaint);

        // Deviation
        y += 30;
        canvas.drawLine(x, y, x + 30, y, deviationPaint);
        canvas.drawText("Deviation >0.5m", x + 40, y + 5, textPaint);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        // Zone drag-edit mode: intercept single-finger drag; pass pinch to zone scale detector
        if (draggableZone != null) {
            // Always feed pinch events to the zone-resize detector
            if (zoneScaleDetector != null) zoneScaleDetector.onTouchEvent(event);

            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN: {
                    float cx = offsetX + draggableZone.x * scale;
                    float cy = offsetY - draggableZone.y * scale;
                    float dist = (float) Math.hypot(event.getX() - cx, event.getY() - cy);
                    if (dist <= 60f) {
                        isDraggingZone = true;
                        autoCenter = false; // prevent the view from snapping back mid-drag
                        return true;        // consume — only when inside hit-target
                    }
                    // Outside hit-target: let gesture detectors see the DOWN normally
                    break;
                }
                case MotionEvent.ACTION_MOVE:
                    if (isDraggingZone && event.getPointerCount() == 1) {
                        // Single finger: move zone position
                        draggableZone.x = (event.getX() - offsetX) / scale;
                        draggableZone.y = (offsetY - event.getY()) / scale;
                        invalidate();
                        return true;
                    } else if (isDraggingZone && event.getPointerCount() > 1) {
                        // 2+ fingers: handled by zoneScaleDetector above — don't also pan
                        return true;
                    }
                    break;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    if (isDraggingZone) {
                        if (zoneDragListener != null)
                            zoneDragListener.onZoneMoved(
                                    draggableZone, draggableZone.x, draggableZone.y);
                        isDraggingZone = false;
                        return true;
                    }
                    break;
            }
        }
        // Normal pan + zoom
        scaleDetector.onTouchEvent(event);
        gestureDetector.onTouchEvent(event);
        return true;
    }

    private class ScaleListener extends ScaleGestureDetector.SimpleOnScaleGestureListener {
        @Override
        public boolean onScale(ScaleGestureDetector detector) {
            scale *= detector.getScaleFactor();
            scale = Math.max(15f, Math.min(scale, 300f));
            autoCenter = false; // Disable auto-center when user zooms
            invalidate();
            return true;
        }
    }

    private class GestureListener extends GestureDetector.SimpleOnGestureListener {
        @Override
        public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY) {
            offsetX -= distanceX;
            offsetY -= distanceY;
            autoCenter = false; // Disable auto-center when user pans
            invalidate();
            return true;
        }
    }
}
