#!/usr/bin/env python3
"""
Training Data Generator for Temperature Prediction Models

This script generates realistic historical sensor data and saves it directly to the
sensors.db database. This data can then be used to train the LSTM models.

Usage:
    python generate_dataset.py                    # Generate default amount of data
    python generate_dataset.py --days 30          # Generate 30 days of data
    python generate_dataset.py --sensors 1 2      # Generate for specific sensors
    python generate_dataset.py --interval 60      # Data point every 60 seconds
"""

import argparse
import math
import random
import time
from datetime import datetime, timedelta
import db


# ===== CONFIGURATION ==========================================================
DEFAULT_DAYS = 7  # Default number of days of historical data
DEFAULT_INTERVAL = 60  # Default interval between readings in seconds
DEFAULT_SENSORS = [1, 2, 3, 4]

# Base temperature for each sensor (in Celsius)
BASE_TEMPS = {
    1: 22.0,  # Room temperature
    2: 24.0,  # Slightly warmer room
    3: 20.0,  # Cooler room
    4: 23.0,  # Another room
}

# Base noise level for each sensor (in dB)
BASE_NOISE = {
    1: 45.0,  # Quiet room
    2: 55.0,  # Normal room
    3: 40.0,  # Very quiet
    4: 50.0,  # Moderate
}


# ===== DATA GENERATION FUNCTIONS ==============================================
def generate_temperature(sensor_id, timestamp, start_time):
    """
    Generate realistic temperature data with:
    - Daily temperature cycles (warmer during day, cooler at night)
    - Weekly patterns (slightly different on weekends)
    - Seasonal trends
    - Random fluctuations
    - Occasional anomalies
    
    Args:
        sensor_id: Sensor identifier
        timestamp: Current timestamp
        start_time: Start timestamp for calculating trends
    
    Returns:
        Temperature value in Celsius
    """
    base_temp = BASE_TEMPS.get(sensor_id, 22.0)
    
    # Time-based calculations
    hours_elapsed = (timestamp - start_time).total_seconds() / 3600
    hour_of_day = timestamp.hour
    day_of_week = timestamp.weekday()
    
    # Daily cycle (warmer during day, cooler at night)
    # Peak around 2 PM (14:00), lowest around 4 AM (04:00)
    daily_cycle = 2.5 * math.sin((hour_of_day - 6) * math.pi / 12)
    
    # Weekly pattern (slightly warmer on weekends due to more activity)
    weekly_pattern = 0.5 if day_of_week >= 5 else 0.0
    
    # Slow drift over time (simulate seasonal changes or HVAC adjustments)
    seasonal_drift = 0.3 * math.sin(hours_elapsed * 0.001)
    
    # Random noise (normal fluctuations)
    noise = random.gauss(0, 0.4)
    
    # Occasional temperature spikes (heating/cooling events, doors opening, etc.)
    spike = 0
    if random.random() < 0.03:  # 3% chance of spike
        spike = random.uniform(-1.5, 2.0)
    
    # Combine all components
    temperature = base_temp + daily_cycle + weekly_pattern + seasonal_drift + noise + spike
    
    return round(temperature, 2)


def generate_noise(sensor_id, timestamp, start_time):
    """
    Generate realistic noise data with:
    - Higher noise during daytime
    - Lower noise at night
    - Random fluctuations
    - Occasional loud events
    
    Args:
        sensor_id: Sensor identifier
        timestamp: Current timestamp
        start_time: Start timestamp for calculating trends
    
    Returns:
        Noise level in dB
    """
    base_noise = BASE_NOISE.get(sensor_id, 50.0)
    
    # Time-based calculations
    hour_of_day = timestamp.hour
    day_of_week = timestamp.weekday()
    
    # Daily pattern (quieter at night, noisier during day)
    if 22 <= hour_of_day or hour_of_day <= 6:
        # Night time (10 PM - 6 AM): much quieter
        daily_pattern = -8.0
    elif 9 <= hour_of_day <= 18:
        # Business hours (9 AM - 6 PM): noisier
        daily_pattern = 5.0
    else:
        # Morning/evening: moderate
        daily_pattern = 0.0
    
    # Weekend pattern (slightly quieter on weekends)
    weekend_pattern = -3.0 if day_of_week >= 5 else 0.0
    
    # Random noise
    noise = random.gauss(0, 2.5)
    
    # Occasional loud events (doors slamming, equipment, conversations, etc.)
    spike = 0
    if random.random() < 0.08:  # 8% chance of noise spike
        spike = random.uniform(5, 20)
    
    # Combine all components
    noise_level = base_noise + daily_pattern + weekend_pattern + noise + spike
    
    # Noise can't be below 30 dB (absolute silence threshold)
    return round(max(30.0, noise_level), 2)


# ===== DATABASE OPERATIONS ====================================================
def generate_and_save_data(sensor_ids, days, interval_seconds, verbose=False):
    """
    Generate historical data and save to database
    
    Args:
        sensor_ids: List of sensor IDs to generate data for
        days: Number of days of historical data to generate
        interval_seconds: Interval between data points in seconds
        verbose: Print detailed progress
    """
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Calculate total data points per sensor
    total_seconds = days * 24 * 3600
    points_per_sensor = total_seconds // interval_seconds
    total_points = points_per_sensor * len(sensor_ids)
    
    print(f"\n{'='*70}")
    print(f"Generating Training Data")
    print(f"{'='*70}")
    print(f"Time range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {days} days")
    print(f"Interval: {interval_seconds} seconds")
    print(f"Sensors: {sensor_ids}")
    print(f"Points per sensor: {points_per_sensor:,}")
    print(f"Total data points: {total_points:,}")
    print(f"{'='*70}\n")
    
    # Generate data
    current_time = start_time
    point_count = 0
    last_progress = 0
    
    start_generation = time.time()
    
    while current_time <= end_time:
        for sensor_id in sensor_ids:
            # Generate temperature and noise
            temperature = generate_temperature(sensor_id, current_time, start_time)
            noise = generate_noise(sensor_id, current_time, start_time)
            
            # Save to database
            timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            db.save_temperature_reading(sensor_id, temperature, timestamp_str)
            db.save_noise_reading(sensor_id, noise, timestamp_str)
            
            point_count += 1
            
            # Progress reporting
            if verbose and point_count % 100 == 0:
                progress = (point_count / total_points) * 100
                print(f"Progress: {progress:.1f}% ({point_count:,}/{total_points:,}) - "
                      f"{current_time.strftime('%Y-%m-%d %H:%M')}")
            elif not verbose:
                # Show progress every 10%
                progress = int((point_count / total_points) * 100)
                if progress >= last_progress + 10:
                    print(f"Progress: {progress}%")
                    last_progress = progress
        
        # Move to next time point
        current_time += timedelta(seconds=interval_seconds)
    
    end_generation = time.time()
    duration = end_generation - start_generation
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"Generation Complete")
    print(f"{'='*70}")
    print(f"Total points generated: {point_count:,}")
    print(f"Time elapsed: {duration:.2f} seconds")
    print(f"Rate: {point_count/duration:.0f} points/second")
    
    # Verify data in database
    print(f"\nVerifying data in database...")
    for sensor_id in sensor_ids:
        data = db.get_recent_temperature_data(sensor_id, limit=10000)
        print(f"  Sensor {sensor_id}: {len(data):,} temperature readings")
    
    print(f"\n✅ Training data generation complete!")
    print(f"{'='*70}\n")


# ===== MAIN SCRIPT ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate training data for temperature prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_dataset.py                      # Generate 7 days of data
  python generate_dataset.py --days 30            # Generate 30 days of data
  python generate_dataset.py --sensors 1 2        # Generate for sensors 1 and 2
  python generate_dataset.py --interval 120       # Data point every 2 minutes
  python generate_dataset.py -d 14 -i 30 -v       # 14 days, 30s interval, verbose
        """
    )
    
    parser.add_argument(
        '-d', '--days',
        type=int,
        default=DEFAULT_DAYS,
        help=f'Number of days of historical data to generate (default: {DEFAULT_DAYS})'
    )
    
    parser.add_argument(
        '-s', '--sensors',
        type=int,
        nargs='+',
        default=DEFAULT_SENSORS,
        help=f'Sensor IDs to generate data for (default: {DEFAULT_SENSORS})'
    )
    
    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=DEFAULT_INTERVAL,
        help=f'Interval between readings in seconds (default: {DEFAULT_INTERVAL})'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed progress during generation'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing data before generating (WARNING: deletes all data!)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.days <= 0:
        print("❌ Error: Days must be positive")
        return
    
    if args.interval <= 0:
        print("❌ Error: Interval must be positive")
        return
    
    if not args.sensors:
        print("❌ Error: At least one sensor ID required")
        return
    
    # Initialize database
    print("🗄️  Initializing database...")
    db.init_database()
    print("✓ Database ready")
    
    # Clear data if requested
    if args.clear:
        print("\n⚠️  WARNING: Clearing all existing data!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() == 'yes':
            import sqlite3
            conn = sqlite3.connect(db.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM temperature_readings")
            cursor.execute("DELETE FROM noise_readings")
            conn.commit()
            conn.close()
            print("✓ Existing data cleared")
        else:
            print("Cancelled - keeping existing data")
    
    # Generate data
    try:
        generate_and_save_data(
            sensor_ids=args.sensors,
            days=args.days,
            interval_seconds=args.interval,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
