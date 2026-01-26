"""
Database monitoring and visualization script.
Helps inspect the stored data and model performance.
"""

import sqlite3
from datetime import datetime, timedelta
import json

DB_PATH = "sensors.db"


def connect_db():
    """Connect to the database"""
    try:
        return sqlite3.connect(DB_PATH)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def get_sensor_summary():
    """Get summary statistics for each sensor"""
    conn = connect_db()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    print("\n" + "=" * 80)
    print("SENSOR SUMMARY")
    print("=" * 80)
    
    # Get all sensors
    cursor.execute("SELECT DISTINCT sensor_id FROM temperature_readings ORDER BY sensor_id")
    sensors = [row[0] for row in cursor.fetchall()]
    
    if not sensors:
        print("No sensor data available yet.")
        conn.close()
        return
    
    for sensor_id in sensors:
        # Get statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as count,
                MIN(temperature) as min_temp,
                MAX(temperature) as max_temp,
                AVG(temperature) as avg_temp,
                MIN(timestamp) as first_reading,
                MAX(timestamp) as last_reading
            FROM temperature_readings
            WHERE sensor_id = ?
        """, (sensor_id,))
        
        row = cursor.fetchone()
        count, min_temp, max_temp, avg_temp, first_reading, last_reading = row
        
        print(f"\n{sensor_id}:")
        print(f"  Total Readings: {count}")
        print(f"  Temperature Range: {min_temp:.2f}°C to {max_temp:.2f}°C")
        print(f"  Average: {avg_temp:.2f}°C")
        print(f"  First Reading: {first_reading}")
        print(f"  Last Reading: {last_reading}")
        
        # Get recent data (last 5 minutes)
        five_min_ago = (datetime.now() - timedelta(minutes=5)).isoformat()
        cursor.execute("""
            SELECT COUNT(*) 
            FROM temperature_readings 
            WHERE sensor_id = ? AND timestamp >= ?
        """, (sensor_id, five_min_ago))
        recent_count = cursor.fetchone()[0]
        print(f"  Last 5 Minutes: {recent_count} readings")
    
    conn.close()
    print("=" * 80)


def get_recent_readings(sensor_id, minutes=5, limit=20):
    """Get recent readings for a sensor"""
    conn = connect_db()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()
    
    cursor.execute("""
        SELECT temperature, timestamp 
        FROM temperature_readings 
        WHERE sensor_id = ? AND timestamp >= ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (sensor_id, cutoff, limit))
    
    readings = cursor.fetchall()
    conn.close()
    
    if not readings:
        print(f"\nNo recent readings for {sensor_id}")
        return
    
    print(f"\n" + "=" * 80)
    print(f"RECENT READINGS - {sensor_id} (Last {minutes} minutes)")
    print("=" * 80)
    print(f"{'Temperature':>12} | {'Timestamp':>25}")
    print("-" * 80)
    
    for temp, timestamp in readings:
        print(f"{temp:>11.2f}°C | {timestamp}")
    
    print("=" * 80)


def get_noise_summary():
    """Get noise sensor summary"""
    conn = connect_db()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as count,
            MIN(noise) as min_noise,
            MAX(noise) as max_noise,
            AVG(noise) as avg_noise,
            MIN(timestamp) as first_reading,
            MAX(timestamp) as last_reading
        FROM noise_readings
    """)
    
    row = cursor.fetchone()
    count, min_noise, max_noise, avg_noise, first_reading, last_reading = row
    
    if count == 0:
        print("\nNo noise data available yet.")
        conn.close()
        return
    
    print("\n" + "=" * 80)
    print("NOISE SENSOR SUMMARY")
    print("=" * 80)
    print(f"Total Readings: {count}")
    print(f"Noise Range: {min_noise:.2f}dB to {max_noise:.2f}dB")
    print(f"Average: {avg_noise:.2f}dB")
    print(f"First Reading: {first_reading}")
    print(f"Last Reading: {last_reading}")
    print("=" * 80)
    
    conn.close()


def get_data_quality_report():
    """Generate a data quality report"""
    conn = connect_db()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    print("\n" + "=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)
    
    # Check for gaps in data
    cursor.execute("SELECT DISTINCT sensor_id FROM temperature_readings")
    sensors = [row[0] for row in cursor.fetchall()]
    
    for sensor_id in sensors:
        cursor.execute("""
            SELECT timestamp 
            FROM temperature_readings 
            WHERE sensor_id = ?
            ORDER BY timestamp
        """, (sensor_id,))
        
        timestamps = [row[0] for row in cursor.fetchall()]
        
        if len(timestamps) < 2:
            continue
        
        # Calculate gaps
        gaps = []
        for i in range(1, len(timestamps)):
            t1 = datetime.fromisoformat(timestamps[i-1])
            t2 = datetime.fromisoformat(timestamps[i])
            gap = (t2 - t1).total_seconds()
            if gap > 5:  # Gap larger than 5 seconds
                gaps.append((t1, t2, gap))
        
        print(f"\n{sensor_id}:")
        print(f"  Total Readings: {len(timestamps)}")
        
        if gaps:
            print(f"  Data Gaps Detected: {len(gaps)}")
            print(f"  Largest Gap: {max(gaps, key=lambda x: x[2])[2]:.1f} seconds")
        else:
            print(f"  Data Continuity: ✓ Good (no gaps > 5 seconds)")
        
        # Check for outliers
        cursor.execute("""
            SELECT 
                AVG(temperature) as avg,
                (AVG(temperature * temperature) - AVG(temperature) * AVG(temperature)) as variance
            FROM temperature_readings
            WHERE sensor_id = ?
        """, (sensor_id,))
        
        avg, variance = cursor.fetchone()
        std = variance ** 0.5 if variance > 0 else 0
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM temperature_readings
            WHERE sensor_id = ? 
            AND (temperature < ? OR temperature > ?)
        """, (sensor_id, avg - 3*std, avg + 3*std))
        
        outliers = cursor.fetchone()[0]
        
        if outliers > 0:
            print(f"  Potential Outliers: {outliers} ({outliers/len(timestamps)*100:.1f}%)")
        else:
            print(f"  Outlier Detection: ✓ None detected")
    
    conn.close()
    print("=" * 80)


def export_sensor_data(sensor_id, output_file=None):
    """Export sensor data to JSON file"""
    conn = connect_db()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT temperature, timestamp 
        FROM temperature_readings 
        WHERE sensor_id = ?
        ORDER BY timestamp
    """, (sensor_id,))
    
    data = [
        {"temperature": row[0], "timestamp": row[1]}
        for row in cursor.fetchall()
    ]
    
    conn.close()
    
    if not output_file:
        output_file = f"{sensor_id}_data.json"
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Exported {len(data)} readings to {output_file}")


def main():
    """Main monitoring interface"""
    print("\n" + "=" * 80)
    print("TEMPERATURE PREDICTION SYSTEM - DATABASE MONITOR")
    print("=" * 80)
    
    while True:
        print("\nMonitoring Options:")
        print("1. Sensor Summary")
        print("2. Recent Readings (specific sensor)")
        print("3. Noise Sensor Summary")
        print("4. Data Quality Report")
        print("5. Export Sensor Data")
        print("6. View All Recent Readings")
        print("0. Exit")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == "0":
            print("\nExiting monitor.\n")
            break
            
        elif choice == "1":
            get_sensor_summary()
            
        elif choice == "2":
            sensor_id = input("Enter sensor ID: ").strip()
            minutes = input("Minutes of history (default 5): ").strip()
            minutes = int(minutes) if minutes else 5
            get_recent_readings(sensor_id, minutes=minutes)
            
        elif choice == "3":
            get_noise_summary()
            
        elif choice == "4":
            get_data_quality_report()
            
        elif choice == "5":
            sensor_id = input("Enter sensor ID: ").strip()
            output_file = input("Output filename (default: {sensor_id}_data.json): ").strip()
            export_sensor_data(sensor_id, output_file if output_file else None)
            
        elif choice == "6":
            conn = connect_db()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT sensor_id FROM temperature_readings")
                sensors = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                for sensor_id in sensors:
                    get_recent_readings(sensor_id, minutes=2, limit=10)
        
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
