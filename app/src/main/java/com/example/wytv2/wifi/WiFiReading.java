package com.example.wytv2.wifi;

/**
 * Represents a single WiFi access point reading.
 */
public class WiFiReading {
    public String bssid; // MAC address of AP
    public String ssid; // Network name
    public int rssi; // Signal strength in dBm
    public long timestamp; // When reading was taken

    public WiFiReading(String bssid, String ssid, int rssi, long timestamp) {
        this.bssid = bssid;
        this.ssid = ssid;
        this.rssi = rssi;
        this.timestamp = timestamp;
    }

    @Override
    public String toString() {
        return String.format("WiFi[%s (%s): %d dBm]", ssid, bssid, rssi);
    }
}
