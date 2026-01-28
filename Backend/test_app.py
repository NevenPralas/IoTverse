#!/usr/bin/env python3
"""
Test script for the temperature prediction Flask app.
Simulates 4 sensors sending temperature readings to trigger training and predictions.
"""

import requests
import time
import numpy as np
import sys

# Configuration
BASE_URL = "http://localhost:8080"
NUM_SENSORS = 4
READINGS_PER_SENSOR = 120  # Need at least 60 for training, 120 gives good data
SEND_INTERVAL = 0.5  # Send every 0.5 seconds (simulating real-time data)

# Sensor configurations with different temperature patterns
SENSOR_CONFIGS = {
    "1": {
        "name": "Office Sensor",
        "base_temp": 22.0,
        "variation": 2.0,
        "trend": 0.01,  # Slight warming trend
        "noise": 0.3
    },
    "2": {
        "name": "Warehouse Sensor",
        "base_temp": 18.0,
        "variation": 3.0,
        "trend": -0.005,  # Slight cooling trend
        "noise": 0.5
    },
    "3": {
        "name": "Server Room Sensor",
        "base_temp": 25.0,
        "variation": 1.0,
        "trend": 0.02,  # Warming trend (AC struggling)
        "noise": 0.2
    },
    "4": {
        "name": "Outdoor Sensor",
        "base_temp": 15.0,
        "variation": 5.0,
        "trend": 0.03,  # Day warming up
        "noise": 0.8
    }
}


def generate_temperature(sensor_id, reading_num):
    """Generate realistic temperature reading for a sensor"""
    config = SENSOR_CONFIGS[sensor_id]
    
    # Base temperature
    temp = config["base_temp"]
    
    # Add trend over time
    temp += config["trend"] * reading_num
    
    # Add sinusoidal variation (simulating cyclical patterns)
    temp += config["variation"] * np.sin(reading_num * 0.1)
    
    # Add random noise
    temp += np.random.normal(0, config["noise"])
    
    return round(temp, 2)


def send_temperature(sensor_id, temperature):
    """Send temperature reading to the Flask app"""
    url = f"{BASE_URL}/sensors/temperature/{sensor_id}"
    payload = {"temperature": temperature}
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending data to sensor {sensor_id}: {e}")
        return False


def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def main():
    print_banner("Temperature Prediction App - Test Script")
    
    # Check if app is running
    print("\n📡 Checking if Flask app is running...")
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=2)
        print(f"✅ App is running! Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Flask app at {BASE_URL}")
        print(f"   Error: {e}")
        print(f"\n   Please make sure the Flask app is running:")
        print(f"   python app.py")
        sys.exit(1)
    
    # Send initial data
    print_banner("Phase 1: Sending Initial Temperature Readings")
    print(f"Sending {READINGS_PER_SENSOR} readings per sensor ({NUM_SENSORS} sensors)")
    print(f"This will take approximately {READINGS_PER_SENSOR * SEND_INTERVAL:.1f} seconds\n")
    
    total_sent = 0
    total_failed = 0
    
    for reading_num in range(READINGS_PER_SENSOR):
        # Send readings for all sensors
        for sensor_id in SENSOR_CONFIGS.keys():
            temp = generate_temperature(sensor_id, reading_num)
            config = SENSOR_CONFIGS[sensor_id]
            
            success = send_temperature(sensor_id, temp)
            
            if success:
                total_sent += 1
                if reading_num % 20 == 0:  # Print progress every 20 readings
                    print(f"📊 Sensor {sensor_id} ({config['name']}): {temp}°C")
            else:
                total_failed += 1
        
        # Progress indicator
        if (reading_num + 1) % 20 == 0:
            progress = (reading_num + 1) / READINGS_PER_SENSOR * 100
            print(f"\n   Progress: {progress:.0f}% ({reading_num + 1}/{READINGS_PER_SENSOR} readings)\n")
        
        time.sleep(SEND_INTERVAL)
    
    print(f"\n✅ Data sending complete!")
    print(f"   Total sent: {total_sent}")
    print(f"   Total failed: {total_failed}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
