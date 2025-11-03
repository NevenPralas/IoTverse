import sqlite3
from datetime import datetime
import pandas as pd

DB_NAME = "iot_data.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def create_table():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            temperature REAL,
            hasSound INTEGER
        )
    """)
    conn.commit()
    conn.close()


def insert_measurement(temperature: float, hasSound: float, time=""):
    if time == "":
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO measurements (time, temperature, hasSound)
        VALUES (?, ?, ?)
    """, (time, temperature, hasSound))
    conn.commit()
    conn.close()


def get_latest_data(limit: int = 100):
    """Returns last N measurements as pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query(
        f"SELECT * FROM measurements ORDER BY id DESC LIMIT {limit}", conn
    )
    conn.close()
    return df


def get_all_data():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM measurements ORDER BY id", conn)
    conn.close()
    return df


def clear_data():
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM measurements")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_table()
    print("Created table 'measurements' in database: ", DB_NAME)
