from database import create_table, insert_measurement, clear_data
from datetime import datetime, timedelta
from sqlite3 import connect
import random

NUM_SAMPLES = 500

clear_data()
create_table()

print(f"Inserting {NUM_SAMPLES} dummy measurements.")

start_time = datetime.now() - timedelta(hours=NUM_SAMPLES / 12)  # each 5 min

for i in range(NUM_SAMPLES):
    temperature = round(20 + random.uniform(-2, 2) + (i / 200), 2)
    hasSound = random.randint(0,1)
    time = (start_time + timedelta(minutes=5*i)).strftime("%Y-%m-%d %H:%M:%S")

    insert_measurement(temperature, hasSound, time)

print("Done inserting!")
