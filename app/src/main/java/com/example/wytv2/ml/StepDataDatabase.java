package com.example.wytv2.ml;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * SQLite database for storing step detection data for continuous learning.
 */
public class StepDataDatabase extends SQLiteOpenHelper {
    private static final String TAG = "StepDataDB";
    private static final String DATABASE_NAME = "step_data.db";
    private static final int DATABASE_VERSION = 1;

    // Table name
    private static final String TABLE_STEPS = "step_data";

    // Column names
    private static final String COL_ID = "id";
    private static final String COL_TIMESTAMP = "timestamp";
    private static final String COL_ACCEL_X = "accel_x";
    private static final String COL_ACCEL_Y = "accel_y";
    private static final String COL_ACCEL_Z = "accel_z";
    private static final String COL_GYRO_X = "gyro_x";
    private static final String COL_GYRO_Y = "gyro_y";
    private static final String COL_GYRO_Z = "gyro_z";
    private static final String COL_MAG_X = "mag_x";
    private static final String COL_MAG_Y = "mag_y";
    private static final String COL_MAG_Z = "mag_z";
    private static final String COL_FEATURES = "features";
    private static final String COL_IS_ACTUAL_STEP = "is_actual_step";
    private static final String COL_THRESHOLD = "threshold";
    private static final String COL_ACTIVITY = "activity_label";
    private static final String COL_CONFIDENCE = "confidence";
    private static final String COL_SOURCE = "source";
    private static final String COL_USED_TRAINING = "used_for_training";

    public StepDataDatabase(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        String createTable = "CREATE TABLE " + TABLE_STEPS + " (" +
                COL_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
                COL_TIMESTAMP + " INTEGER NOT NULL, " +
                COL_ACCEL_X + " REAL, " +
                COL_ACCEL_Y + " REAL, " +
                COL_ACCEL_Z + " REAL, " +
                COL_GYRO_X + " REAL, " +
                COL_GYRO_Y + " REAL, " +
                COL_GYRO_Z + " REAL, " +
                COL_MAG_X + " REAL, " +
                COL_MAG_Y + " REAL, " +
                COL_MAG_Z + " REAL, " +
                COL_FEATURES + " BLOB, " +
                COL_IS_ACTUAL_STEP + " INTEGER, " +
                COL_THRESHOLD + " REAL, " +
                COL_ACTIVITY + " TEXT, " +
                COL_CONFIDENCE + " REAL, " +
                COL_SOURCE + " TEXT, " +
                COL_USED_TRAINING + " INTEGER DEFAULT 0" +
                ")";
        db.execSQL(createTable);

        // Create indices for faster queries
        db.execSQL("CREATE INDEX idx_timestamp ON " + TABLE_STEPS + "(" + COL_TIMESTAMP + ")");
        db.execSQL("CREATE INDEX idx_source ON " + TABLE_STEPS + "(" + COL_SOURCE + ")");
        db.execSQL("CREATE INDEX idx_used_training ON " + TABLE_STEPS + "(" + COL_USED_TRAINING + ")");

        Log.d(TAG, "Database created successfully");
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_STEPS);
        onCreate(db);
    }

    /**
     * Insert a step data point into the database.
     */
    public long insert(StepDataPoint dataPoint) {
        if (!dataPoint.isValid()) {
            Log.e(TAG, "Invalid data point, skipping insert");
            return -1;
        }

        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues values = new ContentValues();

        values.put(COL_TIMESTAMP, dataPoint.timestamp);
        values.put(COL_ACCEL_X, dataPoint.accelerometer[0]);
        values.put(COL_ACCEL_Y, dataPoint.accelerometer[1]);
        values.put(COL_ACCEL_Z, dataPoint.accelerometer[2]);
        values.put(COL_GYRO_X, dataPoint.gyroscope[0]);
        values.put(COL_GYRO_Y, dataPoint.gyroscope[1]);
        values.put(COL_GYRO_Z, dataPoint.gyroscope[2]);
        values.put(COL_MAG_X, dataPoint.magnetometer[0]);
        values.put(COL_MAG_Y, dataPoint.magnetometer[1]);
        values.put(COL_MAG_Z, dataPoint.magnetometer[2]);
        values.put(COL_FEATURES, floatArrayToBytes(dataPoint.features));
        values.put(COL_IS_ACTUAL_STEP, dataPoint.isActualStep ? 1 : 0);
        values.put(COL_THRESHOLD, dataPoint.threshold);
        values.put(COL_ACTIVITY, dataPoint.activityLabel);
        values.put(COL_CONFIDENCE, dataPoint.confidence);
        values.put(COL_SOURCE, dataPoint.source);
        values.put(COL_USED_TRAINING, dataPoint.usedForTraining ? 1 : 0);

        long id = db.insert(TABLE_STEPS, null, values);
        Log.d(TAG, "Inserted data point with ID: " + id);
        return id;
    }

    /**
     * Get total count of collected samples.
     */
    public int getCount() {
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = db.rawQuery("SELECT COUNT(*) FROM " + TABLE_STEPS, null);
        int count = 0;
        if (cursor.moveToFirst()) {
            count = cursor.getInt(0);
        }
        cursor.close();
        return count;
    }

    /**
     * Get count of unused training data.
     */
    public int getUnusedCount() {
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = db.rawQuery(
                "SELECT COUNT(*) FROM " + TABLE_STEPS + " WHERE " + COL_USED_TRAINING + " = 0",
                null);
        int count = 0;
        if (cursor.moveToFirst()) {
            count = cursor.getInt(0);
        }
        cursor.close();
        return count;
    }

    /**
     * Get unused training data (not yet used for training).
     */
    public List<StepDataPoint> getUnusedData(int limit) {
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = db.query(
                TABLE_STEPS,
                null,
                COL_USED_TRAINING + " = 0",
                null, null, null,
                COL_TIMESTAMP + " ASC",
                String.valueOf(limit));

        List<StepDataPoint> dataPoints = new ArrayList<>();
        while (cursor.moveToNext()) {
            dataPoints.add(cursorToDataPoint(cursor));
        }
        cursor.close();

        Log.d(TAG, "Retrieved " + dataPoints.size() + " unused data points");
        return dataPoints;
    }

    /**
     * Mark data points as used for training.
     */
    public void markAsUsed(List<StepDataPoint> dataPoints) {
        SQLiteDatabase db = this.getWritableDatabase();
        db.beginTransaction();
        try {
            for (StepDataPoint point : dataPoints) {
                ContentValues values = new ContentValues();
                values.put(COL_USED_TRAINING, 1);
                db.update(TABLE_STEPS, values,
                        COL_TIMESTAMP + " = ?",
                        new String[] { String.valueOf(point.timestamp) });
            }
            db.setTransactionSuccessful();
            Log.d(TAG, "Marked " + dataPoints.size() + " points as used");
        } finally {
            db.endTransaction();
        }
    }

    /**
     * Clear all data (for testing/privacy).
     */
    public void clearAll() {
        SQLiteDatabase db = this.getWritableDatabase();
        int deleted = db.delete(TABLE_STEPS, null, null);
        Log.d(TAG, "Deleted " + deleted + " records");
    }

    /**
     * Convert cursor to StepDataPoint.
     */
    private StepDataPoint cursorToDataPoint(Cursor cursor) {
        StepDataPoint point = new StepDataPoint();
        point.timestamp = cursor.getLong(cursor.getColumnIndexOrThrow(COL_TIMESTAMP));
        point.accelerometer = new float[] {
                cursor.getFloat(cursor.getColumnIndexOrThrow(COL_ACCEL_X)),
                cursor.getFloat(cursor.getColumnIndexOrThrow(COL_ACCEL_Y)),
                cursor.getFloat(cursor.getColumnIndexOrThrow(COL_ACCEL_Z))
        };
        point.gyroscope = new float[] {
                cursor.getFloat(cursor.getColumnIndexOrThrow(COL_GYRO_X)),
                cursor.getFloat(cursor.getColumnIndexOrThrow(COL_GYRO_Y)),
                cursor.getFloat(cursor.getColumnIndexOrThrow(COL_GYRO_Z))
        };
        point.magnetometer = new float[] {
                cursor.getFloat(cursor.getColumnIndexOrThrow(COL_MAG_X)),
                cursor.getFloat(cursor.getColumnIndexOrThrow(COL_MAG_Y)),
                cursor.getFloat(cursor.getColumnIndexOrThrow(COL_MAG_Z))
        };
        point.features = bytesToFloatArray(
                cursor.getBlob(cursor.getColumnIndexOrThrow(COL_FEATURES)));
        point.isActualStep = cursor.getInt(cursor.getColumnIndexOrThrow(COL_IS_ACTUAL_STEP)) == 1;
        point.threshold = cursor.getFloat(cursor.getColumnIndexOrThrow(COL_THRESHOLD));
        point.activityLabel = cursor.getString(cursor.getColumnIndexOrThrow(COL_ACTIVITY));
        point.confidence = cursor.getFloat(cursor.getColumnIndexOrThrow(COL_CONFIDENCE));
        point.source = cursor.getString(cursor.getColumnIndexOrThrow(COL_SOURCE));
        point.usedForTraining = cursor.getInt(cursor.getColumnIndexOrThrow(COL_USED_TRAINING)) == 1;
        return point;
    }

    /**
     * Convert float array to byte array for BLOB storage.
     */
    private byte[] floatArrayToBytes(float[] array) {
        ByteBuffer buffer = ByteBuffer.allocate(array.length * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        for (float value : array) {
            buffer.putFloat(value);
        }
        return buffer.array();
    }

    /**
     * Convert byte array back to float array.
     */
    private float[] bytesToFloatArray(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        float[] array = new float[bytes.length / 4];
        for (int i = 0; i < array.length; i++) {
            array[i] = buffer.getFloat();
        }
        return array;
    }
}
