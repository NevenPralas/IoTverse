#!/usr/bin/env python3
"""
Temperature Training Data Generator

This script generates training data for temperature sensors only:
- Temperature sensors: readings every 5 seconds

The data includes realistic patterns like:
- Daily temperature cycles
- Weekly patterns (different on weekends)
- Seasonal trends
- Occasional anomalies and spikes

Usage:
    python generate_dataset.py                    # Generate default 7 days
    python generate_dataset.py --days 30          # Generate 30 days
    python generate_dataset.py --sensors 1 2      # Only sensors 1 and 2
    python generate_dataset.py --verbose          # Show detailed progress
"""

import argparse
import math
import random
import time
import sqlite3
from datetime import datetime, timedelta
import sys


# ===== CONFIGURATION ==========================================================
DEFAULT_DAYS = 7
DEFAULT_SENSORS = [1, 2, 3, 4]

# Sensor interval (in seconds)
TEMPERATURE_INTERVAL = 5        # Temperature readings every 5 seconds

# Database configuration
DB_PATH = "sensors.db"

# Base values for each sensor
BASE_TEMPS = {
    1: 22.0,  # Room temperature
    2: 24.0,  # Slightly warmer room
    3: 20.0,  # Cooler room
    4: 23.0,  # Another room
}


# ===== DATABASE SETUP =========================================================
def init_database():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create temperature_readings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS temperature_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id TEXT NOT NULL,
            temperature REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    
    # Create index for better query performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_temp_sensor_time 
        ON temperature_readings(sensor_id, timestamp)
    """)
    
    conn.commit()
    conn.close()
    print("✓ Database initialized")


def clear_existing_data():
    """Clear all existing temperature data from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM temperature_readings")
    
    conn.commit()
    conn.close()
    print("✓ Existing temperature data cleared")


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


# ===== BATCH INSERTION ========================================================
def batch_insert_temperature(conn, data_batch):
    """Insert a batch of temperature readings"""
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO temperature_readings (sensor_id, temperature, timestamp) VALUES (?, ?, ?)",
        data_batch
    )


# ===== MAIN GENERATION FUNCTION ===============================================
def generate_training_data(sensor_ids, days, verbose=False):
    """
    Generate temperature training data with realistic sensor intervals
    
    Args:
        sensor_ids: List of sensor IDs to generate data for
        days: Number of days of historical data to generate
        verbose: Print detailed progress
    """
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Calculate expected data points
    total_seconds = days * 24 * 3600
    temp_points_per_sensor = total_seconds // TEMPERATURE_INTERVAL
    total_temp_points = temp_points_per_sensor * len(sensor_ids)
    
    print(f"\n{'='*70}")
    print(f"Temperature Training Data Generator")
    print(f"{'='*70}")
    print(f"Time range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {days} days")
    print(f"Sensors: {sensor_ids}")
    print(f"\nTemperature sensors:")
    print(f"  Interval: {TEMPERATURE_INTERVAL} seconds")
    print(f"  Points per sensor: {temp_points_per_sensor:,}")
    print(f"  Total points: {total_temp_points:,}")
    print(f"{'='*70}\n")
    
    # Initialize database connection
    conn = sqlite3.connect(DB_PATH)
    
    # Batch configuration
    BATCH_SIZE = 1000
    
    temp_batch = []
    temp_count = 0
    last_progress = 0
    
    start_generation = time.time()
    
    # Generate temperature data (every 5 seconds)
    print("Generating temperature data...")
    current_time = start_time
    
    while current_time <= end_time:
        for sensor_id in sensor_ids:
            temperature = generate_temperature(sensor_id, current_time, start_time)
            timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            
            temp_batch.append((str(sensor_id), temperature, timestamp_str))
            temp_count += 1
            
            # Insert batch when it reaches the batch size
            if len(temp_batch) >= BATCH_SIZE:
                batch_insert_temperature(conn, temp_batch)
                conn.commit()
                temp_batch = []
                
                if verbose:
                    progress = (temp_count / total_temp_points) * 100
                    print(f"  Progress: {progress:.1f}% ({temp_count:,}/{total_temp_points:,}) - "
                          f"{current_time.strftime('%Y-%m-%d %H:%M')}")
                else:
                    progress = int((temp_count / total_temp_points) * 100)
                    if progress >= last_progress + 10:
                        print(f"  Progress: {progress}%")
                        last_progress = progress
        
        current_time += timedelta(seconds=TEMPERATURE_INTERVAL)
    
    # Insert remaining temperature batch
    if temp_batch:
        batch_insert_temperature(conn, temp_batch)
        conn.commit()
    
    conn.close()
    
    end_generation = time.time()
    duration = end_generation - start_generation
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"Generation Complete")
    print(f"{'='*70}")
    print(f"Temperature readings: {temp_count:,}")
    print(f"Time elapsed: {duration:.2f} seconds")
    print(f"Rate: {temp_count/duration:.0f} points/second")
    
    # Verify data in database
    print(f"\nVerifying data in database...")
    verify_data(sensor_ids)
    
    print(f"\n✓ Training data generation complete!")
    print(f"{'='*70}\n")


def verify_data(sensor_ids):
    """Verify the generated data in the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for sensor_id in sensor_ids:
        # Count temperature readings
        cursor.execute(
            "SELECT COUNT(*) FROM temperature_readings WHERE sensor_id = ?",
            (str(sensor_id),)
        )
        temp_count = cursor.fetchone()[0]
        
        # Get date range
        cursor.execute(
            """SELECT MIN(timestamp), MAX(timestamp) 
               FROM temperature_readings 
               WHERE sensor_id = ?""",
            (str(sensor_id),)
        )
        date_range = cursor.fetchone()
        
        print(f"  Sensor {sensor_id}: {temp_count:,} readings")
        if date_range[0] and date_range[1]:
            print(f"    Range: {date_range[0]} to {date_range[1]}")
    
    conn.close()


# ===== MAIN SCRIPT ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate realistic training data for temperature sensors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_dataset.py                      # Generate 7 days of data
  python generate_dataset.py --days 30            # Generate 30 days
  python generate_dataset.py --sensors 1 2        # Only sensors 1 and 2
  python generate_dataset.py -d 14 -v             # 14 days, verbose output
  python generate_dataset.py --clear              # Clear existing data first

Temperature Sensor Interval:
  5 seconds (12 readings per minute, 720 per hour, 17,280 per day per sensor)
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
        '-v', '--verbose',
        action='store_true',
        help='Show detailed progress during generation'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing data before generating (WARNING: deletes all data!)'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing data, do not generate new data'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.days <= 0:
        print("Error: Days must be positive")
        sys.exit(1)
    
    if not args.sensors:
        print("Error: At least one sensor ID required")
        sys.exit(1)
    
    # Initialize database
    print("Initializing database...")
    init_database()
    
    # Verify only mode
    if args.verify_only:
        print("\nVerifying existing data...")
        verify_data(args.sensors)
        print("\nVerification complete!")
        sys.exit(0)
    
    # Clear data if requested
    if args.clear:
        print("\n⚠ WARNING: This will delete all existing temperature data!")
        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() == 'yes':
            clear_existing_data()
        else:
            print("Cancelled - keeping existing data")
            sys.exit(0)
    
    # Generate data
    try:
        generate_training_data(
            sensor_ids=args.sensors,
            days=args.days,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Generation interrupted by user")
        print("Partial data may have been saved to the database")
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
