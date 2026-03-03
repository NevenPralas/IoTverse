import random
from datetime import datetime, timedelta
import db  # Imports your provided db.py file


def generate_sensor_data(num_sensors=4, hours=24, interval_seconds=10):
    """
    Generates synthetic temperature and noise data for multiple sensors at second intervals.
    """
    # Ensure the database and tables are created
    db.init_database()

    sensor_ids = list(range(1, num_sensors + 1))
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)

    print(f"Generating data for {num_sensors} sensors...")
    print(f"Timeframe: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Interval: Every {interval_seconds} seconds")

    current_time = start_time
    records_generated = 0

    while current_time <= end_time:
        # Format timestamp as a string for SQLite DATETIME column
        timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')

        for sensor_id in sensor_ids:
            # --- Temperature Generation ---
            # A mean of 22.0 and stddev of 0.4 keeps ~99% of readings strictly between 20.8°C and 23.2°C
            temperature = round(random.gauss(22.0, 0.4), 2)

            # --- Noise Generation ---
            # Base room noise around 40 dB with slight fluctuations
            noise = round(random.gauss(40.0, 3.0), 2)

            # 5% chance of a noise spike (e.g., someone talking, door closing)
            if random.random() < 0.05:
                noise += round(random.uniform(15.0, 35.0), 2)

            # Save the readings using the functions from db.py
            db.save_temperature_reading(sensor_id, temperature, timestamp_str)
            db.save_noise_reading(sensor_id, noise, timestamp_str)

            records_generated += 2

        # Advance the clock by the specified interval in SECONDS
        current_time += timedelta(seconds=interval_seconds)

    print(f"Successfully generated and inserted {records_generated} total readings!")


if __name__ == "__main__":
    # Generate 1 hour of data at 10-second intervals for 3 sensors
    generate_sensor_data(num_sensors=4, hours=1, interval_seconds=5)