package com.example.wytv2;

import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

// TestActivity.java - For quick validation
public class TestActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Test stationary detection
        testStationaryDetection();

        // Test step detection reset
        testDetectionReset();
    }

    private void testStationaryDetection() {
        // Simulate placing phone on table
        new Handler().postDelayed(() -> {
            Toast.makeText(this,
                    "Place phone on table for stationary test",
                    Toast.LENGTH_LONG).show();
        }, 3000);

        // Simulate picking up and walking
        new Handler().postDelayed(() -> {
            Toast.makeText(this,
                    "Pick up phone and walk 10 steps",
                    Toast.LENGTH_LONG).show();
        }, 10000);
    }

    private void testDetectionReset() {
        // Log when detection resets
        Log.d("PDRTest", "Testing detection reset functionality");
    }
}