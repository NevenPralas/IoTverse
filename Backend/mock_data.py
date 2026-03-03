import time
import threading
import random
import requests
import math

# ===================== CONFIGURATION =====================
# Adjust this to match your actual API's address and port
BASE_URL = "http://localhost:8080"
SENSOR_IDS = [1, 2, 3, 4]

# Intervals in seconds
TEMP_INTERVAL = 5
NOISE_INTERVAL = 2


# ===================== DATA GENERATION =====================
def generate_temperature(sensor_id, counter):
    """
    Generate temperature roughly around 22 degrees Celsius.
    Uses a sine wave based on the counter to simulate AC/Heating cycles.
    """
    base_temp = 22.0

    # Slow oscillation +/- 0.5 degrees to simulate room heating/cooling
    oscillation = math.sin(counter * 0.1) * 0.5

    # Tiny bit of random sensor noise (standard deviation of 0.15)
    sensor_noise = random.gauss(0, 0.15)

    # Add slight offset based on sensor_id so they aren't all perfectly identical
    sensor_offset = (sensor_id * 0.1) - 0.2

    return round(base_temp + oscillation + sensor_noise + sensor_offset, 2)


def generate_noise(sensor_id, counter):
    """
    Generate room noise in dB. Base is around 40 dB with occasional spikes.
    """
    # Standard room background noise
    base_noise = random.gauss(40.0, 3.0)

    # 5% chance of a loud noise spike (door closing, someone talking loud)
    if random.random() < 0.05:
        base_noise += random.uniform(15.0, 35.0)

    # Prevent noise from dipping below realistic absolute minimums for a room (e.g., 25 dB)
    return round(max(25.0, base_noise), 2)


# ===================== NETWORK FUNCTIONS =====================
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


# ===================== THREADING LOOPS =====================
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


# ===================== MAIN EXECUTION =====================
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