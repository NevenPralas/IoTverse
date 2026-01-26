"""
Test script for the temperature prediction system.
Simulates sensor data and verifies the system is working correctly.
"""

import requests
import time
import random
import json
from datetime import datetime

BASE_URL = "http://localhost:8080"

def generate_temperature(sensor_id, base_temp, time_offset):
    """Generate realistic temperature data with slight variations"""
    # Add daily cycle (simple sine wave)
    daily_variation = 3 * random.random() * (time_offset % 86400) / 86400
    # Add random noise
    noise = random.uniform(-0.5, 0.5)
    # Add slow drift
    drift = 0.01 * time_offset
    
    return base_temp + daily_variation + noise + drift


def send_temperature_data(sensor_data):
    """Send temperature data to the API"""
    try:
        response = requests.post(
            f"{BASE_URL}/sensors/temperature",
            json={"temperature": sensor_data},
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending data: {e}")
        return False


def send_noise_data(noise_level):
    """Send noise data to the API"""
    try:
        response = requests.post(
            f"{BASE_URL}/sensors/noisedetector",
            json={"noise": noise_level},
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending noise data: {e}")
        return False


def get_stats():
    """Get system statistics"""
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error getting stats: {e}")
    return None


def get_prediction(sensor_id):
    """Get prediction for a sensor"""
    try:
        response = requests.get(f"{BASE_URL}/api/predict/{sensor_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error getting prediction: {e}")
    return None


def run_data_generation(duration_seconds=300, interval=1):
    """
    Generate and send sensor data for a specified duration.
    
    Args:
        duration_seconds: How long to generate data (default: 5 minutes)
        interval: Seconds between readings (default: 1 second)
    """
    print(f"Starting data generation for {duration_seconds} seconds...")
    print(f"Sending data every {interval} second(s)")
    print("-" * 60)
    
    # Base temperatures for each sensor
    base_temps = {
        "sensor1": 25.0,
        "sensor2": 26.5,
        "sensor3": 24.5,
        "sensor4": 25.5
    }
    
    start_time = time.time()
    count = 0
    
    while time.time() - start_time < duration_seconds:
        count += 1
        time_offset = time.time() - start_time
        
        # Generate temperature for all sensors
        sensor_data = {
            sensor_id: generate_temperature(sensor_id, base_temp, time_offset)
            for sensor_id, base_temp in base_temps.items()
        }
        
        # Generate noise level
        noise_level = 40 + random.uniform(-5, 10)
        
        # Send data
        temp_success = send_temperature_data(sensor_data)
        noise_success = send_noise_data(noise_level)
        
        # Print status
        if count % 10 == 0:  # Print every 10 readings
            status = "✓" if temp_success and noise_success else "✗"
            temps_str = ", ".join([f"{sid}: {temp:.2f}°C" for sid, temp in sensor_data.items()])
            print(f"[{count:4d}] {status} {temps_str}, Noise: {noise_level:.1f}dB")
            
            # Get and print stats every minute
            if count % 60 == 0:
                stats = get_stats()
                if stats:
                    print("\n" + "=" * 60)
                    print("SYSTEM STATISTICS")
                    print("=" * 60)
                    print(f"Temperature readings: {stats.get('temperature_readings', {})}")
                    print(f"Noise readings: {stats.get('noise_readings', 0)}")
                    print(f"Active models: {stats.get('active_models', [])}")
                    print("=" * 60 + "\n")
        
        # Wait for next interval
        time.sleep(interval)
    
    print("\n" + "=" * 60)
    print(f"Data generation completed! Sent {count} readings.")
    print("=" * 60)


def test_predictions():
    """Test prediction functionality"""
    print("\n" + "=" * 60)
    print("TESTING PREDICTIONS")
    print("=" * 60)
    
    sensors = ["sensor1", "sensor2", "sensor3", "sensor4"]
    
    for sensor_id in sensors:
        print(f"\nGetting prediction for {sensor_id}...")
        prediction = get_prediction(sensor_id)
        
        if prediction and "predictions" in prediction:
            preds = prediction["predictions"]
            print(f"  ✓ Received {len(preds)} predictions")
            print(f"  First 5: {[f'{p:.2f}' for p in preds[:5]]}")
            print(f"  Last 5:  {[f'{p:.2f}' for p in preds[-5:]]}")
            print(f"  Range: {min(preds):.2f}°C to {max(preds):.2f}°C")
        else:
            print(f"  ✗ No predictions available (model may need more training data)")
    
    print("=" * 60)


def main():
    """Main test routine"""
    print("\n" + "=" * 60)
    print("TEMPERATURE PREDICTION SYSTEM - TEST SCRIPT")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=2)
        print("✓ Server is running")
    except:
        print("✗ Server is not running!")
        print("\nPlease start the server first:")
        print("  python app.py")
        return
    
    print("\nTest Options:")
    print("1. Generate test data (5 minutes)")
    print("2. Generate test data (custom duration)")
    print("3. Test predictions only")
    print("4. View current statistics")
    print("5. Full test (generate data + test predictions)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        run_data_generation(duration_seconds=300, interval=1)
        print("\nWaiting for models to train...")
        time.sleep(10)
        test_predictions()
        
    elif choice == "2":
        duration = int(input("Enter duration in seconds: "))
        interval = float(input("Enter interval between readings (seconds): "))
        run_data_generation(duration_seconds=duration, interval=interval)
        print("\nWaiting for models to train...")
        time.sleep(10)
        test_predictions()
        
    elif choice == "3":
        test_predictions()
        
    elif choice == "4":
        stats = get_stats()
        if stats:
            print("\n" + "=" * 60)
            print("CURRENT SYSTEM STATISTICS")
            print("=" * 60)
            print(json.dumps(stats, indent=2))
            print("=" * 60)
        else:
            print("Unable to retrieve statistics")
            
    elif choice == "5":
        print("\nRunning full test sequence...")
        run_data_generation(duration_seconds=300, interval=1)
        print("\nWaiting for models to train (30 seconds)...")
        time.sleep(30)
        test_predictions()
        
        stats = get_stats()
        if stats:
            print("\n" + "=" * 60)
            print("FINAL SYSTEM STATISTICS")
            print("=" * 60)
            print(json.dumps(stats, indent=2))
            print("=" * 60)
    
    else:
        print("Invalid choice")
    
    print("\n✓ Test completed!\n")


if __name__ == "__main__":
    main()
