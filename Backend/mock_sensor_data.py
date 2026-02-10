"""
Mock sensor data generator for testing app.py
Sends realistic temperature and noise data to the Flask API
"""

import requests
import time
import random
import math
import threading
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8080"
# BASE_URL = "http://172.20.10.5:8080/"
SENSOR_IDS = [1, 2, 3, 4]

TEMP_INTERVAL = 5      # 5 seconds
NOISE_INTERVAL = 0.2   # 200 milliseconds

BASE_TEMPS = {
    1: 22.0,  # Room temperature
    2: 24.0,  # Slightly warmer room
    3: 20.0,  # Cooler room
    4: 23.0,  # Another room
}

BASE_NOISE = {
    1: 45.0,  # Quiet room
    2: 55.0,  # Normal room
    3: 40.0,  # Very quiet
    4: 50.0,  # Moderate
}


def generate_temperature(sensor_id, counter):
    """
    Generate realistic temperature data with:
    - Slow daily cycle
    - Small random fluctuations
    - Sensor-specific base temperature
    """
    base_temp = BASE_TEMPS.get(sensor_id, 22.0)
    daily_cycle = 2.0 * math.sin(counter * 0.001)
    noise = random.gauss(0, 0.3)
    spike = random.uniform(-1, 1) if random.random() < 0.05 else 0
    temperature = base_temp + daily_cycle + noise + spike
    return round(temperature, 2)


def generate_noise(sensor_id, counter):
    """
    Generate realistic noise data with:
    - Random fluctuations
    - Occasional loud events
    - Sensor-specific base level
    """
    base_noise = BASE_NOISE.get(sensor_id, 50.0)
    noise = random.gauss(0, 2.0)
    spike = random.uniform(5, 15) if random.random() < 0.1 else 0
    noise_level = base_noise + noise + spike
    return round(max(30.0, noise_level), 2)  # Noise can't be below 30 dB


def send_temperature(sensor_id, temperature):
    """Send temperature data to the API"""
    url = f"{BASE_URL}/sensors/temperature/{sensor_id}"
    payload = {"temperature": temperature}

    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"[Temp] Sensor {sensor_id}: {temperature}°C")
        else:
            print(f"[Temp] Sensor {sensor_id} Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[Temp] Sensor {sensor_id} Connection error: {e}")


def send_noise(sensor_id, noise_level):
    """Send noise data to the API"""
    url = f"{BASE_URL}/sensors/noisedetector/{sensor_id}"
    payload = {"noise": noise_level}

    try:
        response = requests.post(url, json=payload, timeout=1)
        if response.status_code == 200:
            print(f"[Noise] Sensor {sensor_id}: {noise_level} dB")
        else:
            print(f"[Noise] Sensor {sensor_id} Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        pass


def run_temperature_loop():
    """Thread function for temperature data"""
    counter = 0
    print(f"-> Temperature simulation started (Interval: {TEMP_INTERVAL}s)")
    while True:
        try:
            for sensor_id in SENSOR_IDS:
                temperature = generate_temperature(sensor_id, counter)
                send_temperature(sensor_id, temperature)

            counter += 1
            time.sleep(TEMP_INTERVAL)
        except Exception as e:
            print(f"Error in temp loop: {e}")


def run_noise_loop():
    """Thread function for noise data"""
    counter = 0
    print(f"-> Noise simulation started (Interval: {NOISE_INTERVAL}s)")
    while True:
        try:
            for sensor_id in SENSOR_IDS:
                noise_level = generate_noise(sensor_id, counter)
                send_noise(sensor_id, noise_level)

            counter += 1
            time.sleep(NOISE_INTERVAL)
        except Exception as e:
            print(f"Error in noise loop: {e}")


def main():
    """Main entry point"""
    print("=" * 60)
    print("Mock Sensor Data Generator (Multi-threaded)")
    print("=" * 60)
    print(f"Target URL: {BASE_URL}")
    print(f"Sensor IDs: {SENSOR_IDS}")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")

    # Create threads
    temp_thread = threading.Thread(target=run_temperature_loop, daemon=True)
    noise_thread = threading.Thread(target=run_noise_loop, daemon=True)

    # Start threads
    temp_thread.start()
    noise_thread.start()

    # Keep the main thread alive to catch KeyboardInterrupt
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Stopping simulation...")
        print("=" * 60)

if __name__ == "__main__":
    main()
