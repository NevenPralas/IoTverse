import sqlite3

DB_PATH = "sensors.db"


# ===================== DATABASE SETUP =====================
def init_database():
    """Initialize the SQLite database with necessary tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create temperature readings table
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS temperature_readings
                   (
                       id
                           INTEGER
                           PRIMARY
                               KEY
                           AUTOINCREMENT,
                       sensor_id
                           INTEGER
                           NOT
                               NULL,
                       temperature
                           REAL
                           NOT
                               NULL,
                       timestamp
                           DATETIME
                           NOT
                               NULL,
                       created_at
                           DATETIME
                           DEFAULT
                               CURRENT_TIMESTAMP
                   )
                   """)

    # Create noise readings table
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS noise_readings
                   (
                       id
                           INTEGER
                           PRIMARY
                               KEY
                           AUTOINCREMENT,
                       sensor_id
                           INTEGER
                           NOT
                               NULL,
                       noise
                           REAL
                           NOT
                               NULL,
                       timestamp
                           DATETIME
                           NOT
                               NULL,
                       created_at
                           DATETIME
                           DEFAULT
                               CURRENT_TIMESTAMP
                   )
                   """)

    # Create index for faster queries
    cursor.execute("""
                   CREATE INDEX IF NOT EXISTS idx_temp_sensor_time
                       ON temperature_readings (sensor_id, timestamp)
                   """)

    conn.commit()
    conn.close()
    print("Database initialized successfully")


def save_temperature_reading(sensor_id, temperature, timestamp):
    """Save temperature reading to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO temperature_readings (sensor_id, temperature, timestamp) VALUES (?, ?, ?)",
        (sensor_id, temperature, timestamp)
    )
    conn.commit()
    conn.close()


def save_noise_reading(sensor_id, noise, timestamp):
    """Save noise reading to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO noise_readings (sensor_id, noise, timestamp) VALUES (?, ?, ?)",
        (sensor_id, noise, timestamp)
    )
    conn.commit()
    conn.close()


def get_recent_temperature_data(sensor_id, limit=60):
    """Get recent temperature data for a sensor"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT temperature, timestamp
                   FROM temperature_readings
                   WHERE sensor_id = ?
                   ORDER BY timestamp ASC
                   LIMIT ?
                   """, (sensor_id, limit))

    data = cursor.fetchall()
    conn.close()
    return data


def get_unique_sensor_ids():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT sensor_id FROM temperature_readings")
    sensor_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return sensor_ids
