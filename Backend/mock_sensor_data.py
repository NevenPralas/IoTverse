"""
Mock sensor data generator for testing app.py
Sends realistic temperature and noise data to the Flask API
"""

import requests
import time
import random
import math
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8080"
SENSOR_IDS = [1, 2, 3, 4]
SEND_INTERVAL = 1

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
            print(f"✓ Sensor {sensor_id} - Temperature: {temperature}°C")
        else:
            print(f"✗ Sensor {sensor_id} - Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Sensor {sensor_id} - Connection error: {e}")


def send_noise(sensor_id, noise_level):
    """Send noise data to the API"""
    url = f"{BASE_URL}/sensors/noisedetector/{sensor_id}"
    payload = {"noise": noise_level}
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"✓ Sensor {sensor_id} - Noise: {noise_level} dB")
        else:
            print(f"✗ Sensor {sensor_id} - Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Sensor {sensor_id} - Connection error: {e}")


def main():
    """Main loop to continuously send mock sensor data"""
    print("=" * 60)
    print("Mock Sensor Data Generator")
    print("=" * 60)
    print(f"Target URL: {BASE_URL}")
    print(f"Sensor IDs: {SENSOR_IDS}")
    print(f"Send interval: {SEND_INTERVAL} second(s)")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")
    
    counter = 0
    
    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] Sending data (iteration {counter})...")
            
            for sensor_id in SENSOR_IDS:
                temperature = generate_temperature(sensor_id, counter)
                send_temperature(sensor_id, temperature)
                
                noise_level = generate_noise(sensor_id, counter)
                send_noise(sensor_id, noise_level)
            
            counter += 1
            time.sleep(SEND_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print(f"Stopped after {counter} iterations")
        print("=" * 60)


if __name__ == "__main__":
    main()
