import os
from flask import Flask, request, jsonify
import requests
import datetime
from dotenv import load_dotenv
import threading
import time
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import json
import db
import urllib3

# Suppress InsecureRequestWarning for unverified HTTPS requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

app = Flask(__name__)

DATA_JEDI_URL = "https://djx.entlab.hr/m2m/trusted/data"
MODELS_DIR = "models"

SENSOR_IDS = [1, 2, 3, 4]

INITIAL_TRAINING_DELAY_SEC = 20
TRAINING_SCHEDULE_SEC = 120
REQ_DATA_POINTS = 60
SEQUENCE_SIZE = 30

HEADERS = {
    "Authorization": "PREAUTHENTICATED",
    "X-Requester-Id": "digiphy1",
    "X-Requester-Type": "domainApplication",
    "Content-Type": "application/vnd.ericsson.m2m.input+json;version=1.0"
}

# Global model storage for each sensor
models = {}
model_lock = threading.Lock()

# Prediction queues for each sensor - stores the rolling window of predictions
prediction_queues = {}
queue_lock = threading.Lock()

# Track which sensors have been initialized (for knowing when to send all 30 predictions)
initialized_sensors = set()
init_lock = threading.Lock()


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
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


def ensure_models_directory():
    """Ensure the models directory exists"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created models directory: {MODELS_DIR}")


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
        X.append(temperatures_norm[i:i + seq_length])
        y.append(temperatures_norm[i + seq_length])

    return np.array(X), np.array(y), mean, std


def save_model_to_disk(sensor_id):
    """Save the model and its metadata to disk"""
    with model_lock:
        if sensor_id not in models:
            print(f"No model to save for sensor {sensor_id}")
            return False

        model_info = models[sensor_id]
        model = model_info['model']
        mean = model_info['mean']
        std = model_info['std']

    try:
        # Save model state dict
        model_path = os.path.join(MODELS_DIR, f"model_{sensor_id}.pt")
        torch.save(model.state_dict(), model_path)

        # Save metadata (mean, std) - removed prediction_sequence
        metadata_path = os.path.join(MODELS_DIR, f"metadata_{sensor_id}.json")
        metadata = {
            'mean': float(mean),
            'std': float(std),
            'saved_at': datetime.datetime.now(datetime.UTC).isoformat()
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        print(f"Model saved to disk for sensor {sensor_id}")
        return True
    except Exception as e:
        print(f"Error saving model for sensor {sensor_id}: {e}")
        return False


def load_model_from_disk(sensor_id):
    """Load the model and its metadata from disk"""
    model_path = os.path.join(MODELS_DIR, f"model_{sensor_id}.pt")
    metadata_path = os.path.join(MODELS_DIR, f"metadata_{sensor_id}.json")

    # Check if both files exist
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print(f"No saved model found for sensor {sensor_id}")
        return False

    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Create new model instance
        model = TemperatureLSTM(input_size=1, hidden_size=64, num_layers=2)

        # Load model weights
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # Store in global models dict (removed prediction_sequence)
        with model_lock:
            models[sensor_id] = {
                'model': model,
                'mean': metadata['mean'],
                'std': metadata['std']
            }

        print(f"Model loaded from disk for sensor {sensor_id} (saved at {metadata.get('saved_at', 'unknown')})")
        return True
    except Exception as e:
        print(f"Error loading model for sensor {sensor_id}: {e}")
        return False


def load_all_models_from_disk():
    """Load all saved models from disk on startup"""
    if not os.path.exists(MODELS_DIR):
        print("No models directory found")
        return

    # Find all model files
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('model_') and f.endswith('.pt')]

    if not model_files:
        print("No saved models found")
        return

    print(f"Found {len(model_files)} saved model(s)")

    for model_file in model_files:
        # Extract sensor_id from filename (e.g., "model_123.pt" -> "123")
        sensor_id = model_file.replace('model_', '').replace('.pt', '')
        load_model_from_disk(sensor_id)


def train_model(sensor_id):
    """Train or update the model for a specific sensor"""
    print(f"Training model for sensor: {sensor_id}")

    data = db.get_recent_temperature_data(sensor_id)

    if len(data) < REQ_DATA_POINTS:
        print(f"Not enough data for sensor {sensor_id}: {len(data)} points")
        return

    seq_length = SEQUENCE_SIZE

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
            batch_X = X_tensor[i:i + batch_size]
            batch_y = y_tensor[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"Sensor {sensor_id} - Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

    print(f"Model training completed for sensor: {sensor_id}")

    # Save model to disk after training
    save_model_to_disk(sensor_id)


def initialize_prediction_queue(sensor_id):
    """Initialize prediction queue with 30 initial predictions"""
    print(f"Initializing prediction queue for sensor {sensor_id}...")

    with model_lock:
        if sensor_id not in models:
            print(f"No model available for sensor {sensor_id}")
            return False

        model_info = models[sensor_id]
        model = model_info['model']
        mean = model_info['mean']
        std = model_info['std']

    data = db.get_recent_temperature_data(sensor_id)

    if len(data) < 30:
        print(f"Not enough recent data for prediction: {len(data)} points")
        return False

    # Use last 30 readings as input
    temperatures = np.array([d[0] for d in data[-30:]])
    temperatures_norm = (temperatures - mean) / std

    model.eval()
    predictions = []

    # Predict next 30 seconds (one at a time, using previous predictions)
    # NOTE: This is only for initial setup - incremental predictions will use real data
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
    predictions_denorm = (np.array(predictions) * std + mean).tolist()

    # Initialize the queue
    with queue_lock:
        prediction_queues[sensor_id] = deque(predictions_denorm, maxlen=30)

    # Mark this sensor as newly initialized (will send all 30 predictions on first update)
    with init_lock:
        initialized_sensors.add(sensor_id)

    print(f"Initialized prediction queue for sensor {sensor_id} with {len(predictions_denorm)} predictions")
    return True


def predict_next_single_value(sensor_id):
    """Predict only the next single value using the data from the database."""
    with model_lock:
        if sensor_id not in models:
            print(f"No model available for sensor {sensor_id}")
            return None

        model_info = models[sensor_id]
        model = model_info['model']
        mean = model_info['mean']
        std = model_info['std']

    # CRITICAL FIX: Get REAL data from database instead of using prediction_sequence
    data = db.get_recent_temperature_data(sensor_id)

    if len(data) < 30:
        print(f"Not enough recent data for prediction: {len(data)} points")
        return None

    # Use the last 30 REAL temperature readings
    temperatures = np.array([d[0] for d in data[-30:]])
    temperatures_norm = (temperatures - mean) / std

    model.eval()

    with torch.no_grad():
        # Use the last 30 REAL values from the database
        X = torch.FloatTensor(temperatures_norm[-30:]).unsqueeze(0).unsqueeze(-1)

        # Predict next value
        pred_norm = model(X).item()

        # Denormalize
        pred_denorm = pred_norm * std + mean

    return pred_denorm


def update_prediction_queue(sensor_id):
    """Update the prediction queue by removing oldest and adding newest prediction"""
    new_prediction = predict_next_single_value(sensor_id)

    if new_prediction is None:
        return None

    with queue_lock:
        if sensor_id not in prediction_queues:
            print(f"Queue not initialized for sensor {sensor_id}")
            return None

        # Add new prediction (automatically removes oldest due to maxlen=30)
        prediction_queues[sensor_id].append(new_prediction)

        # Return current queue as list
        return list(prediction_queues[sensor_id])


def get_current_predictions(sensor_id):
    """Get the current 30 predictions from the queue"""
    with queue_lock:
        if sensor_id not in prediction_queues:
            return None
        return list(prediction_queues[sensor_id])


def send_predictions_to_datajedi(sensor_id, predictions, send_all=False):
    """Send predicted temperature readings to DataJediX"""
    if predictions is None or len(predictions) == 0:
        return

    current_time = datetime.datetime.now(datetime.UTC).replace(microsecond=0)

    if send_all:
        predictions_to_send = list(enumerate(predictions))
    else:
        predictions_to_send = [(29, predictions[-1])]

    for i, temp in predictions_to_send:
        future_time = current_time + datetime.timedelta(seconds=i + 1)

        # TODO: define proper temperature prediction resource on the platform
        payload = {
            "source": {
                "operator": os.getenv("OPERATOR_ID"),
                "domainApplication": os.getenv("DOMAIN_APP_ID"),
                "user": os.getenv("USER_ID"),
                "res": f"dipProj25_temperature_prediction{sensor_id}"
            },
            "contentNodes": [
                {
                    "value": float(temp),
                    "time": future_time.isoformat()
                }
            ]
        }

        r = requests.post(DATA_JEDI_URL, json=payload, headers=HEADERS, verify=False)
        # Don't use jsonify outside Flask request context
        print(f"Sent prediction for sensor {sensor_id}: {temp}")


# ===================== BACKGROUND TASKS =====================
def training_loop():
    """Background thread that periodically trains models"""
    print("Training loop started")
    # time.sleep(30)  # Wait for some initial data

    while True:
        try:
            sensor_ids = SENSOR_IDS

            # Train model for each sensor
            for sensor_id in sensor_ids:
                print(f"Starting training for sensor {sensor_id}...")
                train_model(sensor_id)
                print(f"Training completed for sensor {sensor_id}")
                # Note: Prediction queue initialization is handled by prediction_loop

            print("Training cycle completed. Training again in 2 minutes...")
            time.sleep(120)  # Train every 2 minutes

        except Exception as e:
            print(f"Error in training loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)


def generate_and_send_prediction(sensor_id):
    """Generate a prediction and send it to DataJediX when temperature reading is received"""
    try:
        # Ensure queue is initialized
        queue_exists = False
        with queue_lock:
            queue_exists = sensor_id in prediction_queues

        if not queue_exists:
            print(f"Initializing queue for sensor {sensor_id}...")
            if not initialize_prediction_queue(sensor_id):
                print(f"Failed to initialize queue for sensor {sensor_id}")
                return

        # Update queue with 1 new prediction (using REAL data from database)
        predictions = update_prediction_queue(sensor_id)

        if predictions and len(predictions) == 30:
            # Check if this sensor was just initialized
            is_first_send = False
            with init_lock:
                if sensor_id in initialized_sensors:
                    is_first_send = True
                    initialized_sensors.remove(sensor_id)

            # Send predictions: all 30 on first send, only last one afterward
            send_predictions_to_datajedi(sensor_id, predictions, send_all=is_first_send)
        else:
            print(f"Not enough predictions for sensor {sensor_id}: {len(predictions) if predictions else 0}")

    except Exception as e:
        print(f"Error generating prediction for sensor {sensor_id}: {e}")
        import traceback
        traceback.print_exc()


# ===================== FLASK ROUTES =====================
@app.route("/sensors/temperature/<sensor_id>", methods=["POST"])
def receive_temperature(sensor_id):
    data = request.get_json()
    print(f"🌡️Received temperature from sensor {sensor_id}: {data}")
    temp_value = data.get("temperature")

    if temp_value is None:
        return jsonify({"error": "Missing temperature value"}), 400

    timestamp = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()
    db.save_temperature_reading(sensor_id, temp_value, timestamp)

    # Send to DataJediX
    payload = {
        "source": {
            "operator": os.getenv("OPERATOR_ID"),
            "domainApplication": os.getenv("DOMAIN_APP_ID"),
            "user": os.getenv("USER_ID"),
            "res": f"dipProj25_temperature{sensor_id}"
        },
        "contentNodes": [
            {
                "value": temp_value,
                "time": timestamp
            }
        ]
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=HEADERS, verify=False)
    
    # Generate and send prediction after receiving temperature reading
    generate_and_send_prediction(sensor_id)
    
    return jsonify({"status": "ok", "platform_code": r.status_code})


@app.route("/sensors/noisedetector/<sensor_id>", methods=["POST"])
def receive_noise(sensor_id):
    data = request.get_json()
    print(f"🗣Received noise from {sensor_id}: {data}")
    noise_value = data["noise"]

    timestamp = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()
    db.save_noise_reading(sensor_id, noise_value, timestamp)

    payload = {
        "source": {
            "operator": os.getenv("OPERATOR_ID"),
            "domainApplication": os.getenv("DOMAIN_APP_ID"),
            "user": os.getenv("USER_ID"),
            "resource": f"dipProj25_noise_detector{sensor_id}"
        },
        "contentNodes": [
            {
                "value": noise_value,
                "time": timestamp
            }
        ]
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=HEADERS, verify=False)
    return jsonify({"status": "ok", "platform_code": r.status_code})


if __name__ == "__main__":
    ensure_models_directory()
    db.init_database()

    print("Loading saved models from disk...")
    load_all_models_from_disk()

    print("Starting training thread...")
    training_thread = threading.Thread(target=training_loop, daemon=True)
    training_thread.start()

    print("Server ready - predictions will be generated on-demand when temperature readings are received")
    app.run(host="0.0.0.0", port=8080, debug=False)
