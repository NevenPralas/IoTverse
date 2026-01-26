import os
from flask import Flask, request, jsonify
import requests
import datetime
from dotenv import load_dotenv
import sqlite3
import threading
import time
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

load_dotenv()

app = Flask(__name__)

DATA_JEDI_URL = "https://djx.entlab.hr/m2m/trusted/data"
DB_PATH = "sensors.db"

# Global model storage for each sensor
models = {}
model_lock = threading.Lock()


# ===================== DATABASE SETUP =====================
def init_database():
    """Initialize the SQLite database with necessary tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create temperature readings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS temperature_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id TEXT NOT NULL,
            temperature REAL NOT NULL,
            timestamp DATETIME NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create noise readings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS noise_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            noise REAL NOT NULL,
            timestamp DATETIME NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_temp_sensor_time 
        ON temperature_readings(sensor_id, timestamp)
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


def save_noise_reading(noise, timestamp):
    """Save noise reading to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO noise_readings (noise, timestamp) VALUES (?, ?)",
        (noise, timestamp)
    )
    conn.commit()
    conn.close()


def get_recent_temperature_data(sensor_id, minutes=10):
    """Get recent temperature data for a sensor"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get data from last N minutes
    cutoff_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=minutes)
    
    cursor.execute("""
        SELECT temperature, timestamp 
        FROM temperature_readings 
        WHERE sensor_id = ? AND timestamp >= ?
        ORDER BY timestamp ASC
    """, (sensor_id, cutoff_time.isoformat()))
    
    data = cursor.fetchall()
    conn.close()
    return data


# ===================== PYTORCH MODEL =====================
class TemperatureLSTM(nn.Module):
    """LSTM model for temperature time series prediction"""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(TemperatureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


def prepare_sequences(data, seq_length=30):
    """Prepare sequences for training"""
    if len(data) < seq_length + 1:
        return None, None
    
    temperatures = np.array([d[0] for d in data])
    
    # Normalize data
    mean = temperatures.mean()
    std = temperatures.std() if temperatures.std() > 0 else 1.0
    temperatures_norm = (temperatures - mean) / std
    
    X, y = [], []
    for i in range(len(temperatures_norm) - seq_length):
        X.append(temperatures_norm[i:i+seq_length])
        y.append(temperatures_norm[i+seq_length])
    
    return np.array(X), np.array(y), mean, std


def train_model(sensor_id):
    """Train or update the model for a specific sensor"""
    print(f"Training model for sensor: {sensor_id}")
    
    # Get training data
    data = get_recent_temperature_data(sensor_id, minutes=30)
    
    if len(data) < 60:  # Need at least 60 data points
        print(f"Not enough data for sensor {sensor_id}: {len(data)} points")
        return
    
    # Prepare sequences
    seq_length = 30
    result = prepare_sequences(data, seq_length)
    if result is None:
        print(f"Cannot prepare sequences for sensor {sensor_id}")
        return
    
    X, y, mean, std = result
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dimension
    y_tensor = torch.FloatTensor(y).unsqueeze(-1)
    
    # Initialize or get existing model
    with model_lock:
        if sensor_id not in models:
            models[sensor_id] = {
                'model': TemperatureLSTM(input_size=1, hidden_size=64, num_layers=2),
                'mean': mean,
                'std': std
            }
        else:
            models[sensor_id]['mean'] = mean
            models[sensor_id]['std'] = std
        
        model = models[sensor_id]['model']
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model.train()
    epochs = 50
    batch_size = 16
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"Sensor {sensor_id} - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    print(f"Model training completed for sensor: {sensor_id}")


def predict_next_30_seconds(sensor_id):
    """Predict the next 30 temperature readings (1 per second)"""
    with model_lock:
        if sensor_id not in models:
            print(f"No model available for sensor {sensor_id}")
            return None
        
        model_info = models[sensor_id]
        model = model_info['model']
        mean = model_info['mean']
        std = model_info['std']
    
    # Get recent data
    data = get_recent_temperature_data(sensor_id, minutes=2)
    
    if len(data) < 30:
        print(f"Not enough recent data for prediction: {len(data)} points")
        return None
    
    # Use last 30 readings as input
    temperatures = np.array([d[0] for d in data[-30:]])
    temperatures_norm = (temperatures - mean) / std
    
    model.eval()
    predictions = []
    
    # Predict next 30 seconds (one at a time, using previous predictions)
    current_sequence = temperatures_norm.copy()
    
    with torch.no_grad():
        for _ in range(30):
            # Prepare input
            X = torch.FloatTensor(current_sequence[-30:]).unsqueeze(0).unsqueeze(-1)
            
            # Predict
            pred_norm = model(X).item()
            predictions.append(pred_norm)
            
            # Add prediction to sequence for next iteration
            current_sequence = np.append(current_sequence, pred_norm)
    
    # Denormalize predictions
    predictions_denorm = np.array(predictions) * std + mean
    
    return predictions_denorm.tolist()


def send_predictions_to_datajedi(sensor_id, predictions):
    """Send predicted temperature readings to DataJediX"""
    if predictions is None or len(predictions) == 0:
        return
    
    headers = {
        "Authorization": "PREAUTHENTICATED",
        "X-Requester-Id": "digiphy1",
        "X-Requester-Type": "domainApplication",
        "Content-Type": "application/vnd.ericsson.m2m.input+json;version=1.0"
    }
    
    current_time = datetime.datetime.now(datetime.UTC)
    
    # Send each prediction with 1-second intervals
    for i, temp in enumerate(predictions):
        future_time = current_time + datetime.timedelta(seconds=i+1)
        
        payload = {
            "source": {
                "operator": os.getenv("OPERATOR_ID"),
                "domainApplication": os.getenv("DOMAIN_APP_ID"),
                "user": os.getenv("USER_ID"),
                "resourceSpec": f"temperature_prediction_{sensor_id}"
            },
            "contentNodes": [
                {
                    "value": float(temp),
                    "time": future_time.isoformat()
                }
            ]
        }
        
        try:
            r = requests.post(DATA_JEDI_URL, json=payload, headers=headers, verify=False)
            print(f"Prediction sent for {sensor_id} at +{i+1}s: {temp:.2f}°C, Status: {r.status_code}")
        except Exception as e:
            print(f"Error sending prediction: {e}")


# ===================== BACKGROUND TASKS =====================
def training_loop():
    """Background thread that periodically trains models"""
    print("Training loop started")
    time.sleep(30)  # Wait for some initial data
    
    while True:
        try:
            # Get all unique sensor IDs
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT sensor_id FROM temperature_readings")
            sensor_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            # Train model for each sensor
            for sensor_id in sensor_ids:
                train_model(sensor_id)
            
            print("Training cycle completed. Waiting 2 minutes...")
            time.sleep(120)  # Train every 2 minutes
            
        except Exception as e:
            print(f"Error in training loop: {e}")
            time.sleep(60)


def prediction_loop():
    """Background thread that periodically makes and sends predictions"""
    print("Prediction loop started")
    time.sleep(60)  # Wait for initial training
    
    while True:
        try:
            # Get all unique sensor IDs
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT sensor_id FROM temperature_readings")
            sensor_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            # Make and send predictions for each sensor
            for sensor_id in sensor_ids:
                predictions = predict_next_30_seconds(sensor_id)
                if predictions:
                    send_predictions_to_datajedi(sensor_id, predictions)
            
            print("Prediction cycle completed. Waiting 30 seconds...")
            time.sleep(30)  # Predict every 30 seconds
            
        except Exception as e:
            print(f"Error in prediction loop: {e}")
            time.sleep(30)


# ===================== FLASK ROUTES =====================
@app.route("/sensors/temperature", methods=["POST"])
def receive_temperature():
    data = request.get_json()
    print(f"Received: {data}")
    
    # Extract sensor ID and temperature
    # Assuming data format: {"sensor_id": "sensor1", "temperature": 25.5}
    # Or if temperature is a dict: {"temperature": {"sensor1": 25.5, "sensor2": 26.0}}
    
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    
    # Handle different data formats
    if isinstance(data.get("temperature"), dict):
        # Multiple sensors in one request
        for sensor_id, temp in data["temperature"].items():
            save_temperature_reading(sensor_id, temp, timestamp)
    else:
        # Single sensor
        sensor_id = data.get("sensor_id", "default_sensor")
        temperature = data["temperature"]
        save_temperature_reading(sensor_id, temperature, timestamp)
    
    # Send to DataJediX
    payload = {
        "source": {
            "operator": os.getenv("OPERATOR_ID"),
            "domainApplication": os.getenv("DOMAIN_APP_ID"),
            "user": os.getenv("USER_ID"),
            "resourceSpec": "temperature"
        },
        "contentNodes": [
            {
                "value": data["temperature"],
                "time": timestamp
            }
        ]
    }

    headers = {
        "Authorization": "PREAUTHENTICATED",
        "X-Requester-Id": "digiphy1",
        "X-Requester-Type": "domainApplication",
        "Content-Type": "application/vnd.ericsson.m2m.input+json;version=1.0"
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=headers, verify=False)
    print("Data Jedi response:", r.status_code, r.text)
    return jsonify({"status": "ok", "platform_code": r.status_code})


@app.route("/sensors/noisedetector", methods=["POST"])
def receive_noise():
    data = request.get_json()
    print(f"Received noise: {data}")
    
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    save_noise_reading(data["noise"], timestamp)

    payload = {
        "source": {
            "operator": os.getenv("OPERATOR_ID"),
            "domainApplication": os.getenv("DOMAIN_APP_ID"),
            "user": os.getenv("USER_ID"),
            "resource": "dipProj25_noise_detector"
        },
        "contentNodes": [
            {
                "value": data["noise"],
                "time": timestamp
            }
        ]
    }

    headers = {
        "Authorization": "PREAUTHENTICATED",
        "X-Requester-Id": "digiphy1",
        "X-Requester-Type": "domainApplication",
        "Content-Type": "application/vnd.ericsson.m2m.input+json;version=1.0"
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=headers, verify=False)
    return jsonify({"status": "ok", "platform_code": r.status_code})


@app.route("/api/predict/<sensor_id>", methods=["GET"])
def get_prediction(sensor_id):
    """API endpoint to manually trigger prediction for a sensor"""
    predictions = predict_next_30_seconds(sensor_id)
    if predictions:
        return jsonify({"sensor_id": sensor_id, "predictions": predictions})
    else:
        return jsonify({"error": "Unable to generate predictions"}), 400


@app.route("/api/train/<sensor_id>", methods=["POST"])
def trigger_training(sensor_id):
    """API endpoint to manually trigger training for a sensor"""
    threading.Thread(target=train_model, args=(sensor_id,)).start()
    return jsonify({"status": "training_started", "sensor_id": sensor_id})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get database statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT sensor_id, COUNT(*) FROM temperature_readings GROUP BY sensor_id")
    temp_stats = {row[0]: row[1] for row in cursor.fetchall()}
    
    cursor.execute("SELECT COUNT(*) FROM noise_readings")
    noise_count = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        "temperature_readings": temp_stats,
        "noise_readings": noise_count,
        "active_models": list(models.keys())
    })


if __name__ == "__main__":
    # Initialize database
    init_database()
    
    # Start background threads
    training_thread = threading.Thread(target=training_loop, daemon=True)
    training_thread.start()
    
    prediction_thread = threading.Thread(target=prediction_loop, daemon=True)
    prediction_thread.start()
    
    # Run Flask app
    app.run(host="0.0.0.0", port=8080, debug=False)
