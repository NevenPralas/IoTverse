import os
import random

from flask import Flask, request, jsonify
import requests
import datetime
from dotenv import load_dotenv
import threading
import time
import torch
import torch.nn as nn
import numpy as np
import json
import db
import urllib3

from threading import Thread


# ===== CONSTANTS ==============================================================
DATA_JEDI_URL = "https://djx.entlab.hr/m2m/trusted/data"
HEADERS = {
    "Authorization": "PREAUTHENTICATED",
    "X-Requester-Id": "digiphy1",
    "X-Requester-Type": "domainApplication",
    "Content-Type": "application/vnd.ericsson.m2m.input+json;version=1.0"
}
MODELS_DIR = "models"
SENSOR_IDS = [1, 2, 3, 4]
TRAINING_SCHEDULE_SEC = 300  # Retrain every x minutes

REQ_DATA_POINTS = 200
SEQUENCE_SIZE = 60
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.2

# FIXED: Training data strategy
# We want to use ALL historical data but give more weight to recent data
RECENT_DATA_WINDOW_SEC = 120  # Last 2 minutes (recent data)
HISTORICAL_DATA_LIMIT = 5000   # Use up to 5000 historical points
RECENT_DATA_WEIGHT = 3.0       # Recent data is weighted 3x more during training


# ===== CONFIG =================================================================
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
app = Flask(__name__)


# ===== GLOBALS ================================================================
models = {}
# model_lock = threading.Lock()


def createIsoStringTimestampGMT1():
    return (datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)).isoformat()


# ===== PYTORCH MODEL ==========================================================
class TemperatureLSTM(nn.Module):
    """Enhanced LSTM model for temperature time series prediction"""

    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.2, output_size=1):
        super(TemperatureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = self.dropout(last_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        prediction = self.fc2(x)
        return prediction


def ensure_models_directory():
    """Ensure the models directory exists"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"✓ Created models directory: {MODELS_DIR}")


# ===== FIXED: PREPARATION WITH ALL NORMALIZATION PARAMS ======================
def prepare_sequences(data, seq_length=60):
    """
    FIXED: Returns ALL normalization parameters
    """
    if len(data) < seq_length + 1:
        return None

    temperatures = np.array([d[0] for d in data])

    # Calculate rate of change
    temp_diff = np.diff(temperatures, prepend=temperatures[0])
    
    # Calculate moving averages
    window_size = min(7, len(temperatures) // 4)
    if window_size > 1:
        moving_avg = np.convolve(temperatures, np.ones(window_size)/window_size, mode='same')
    else:
        moving_avg = temperatures.copy()

    # Calculate normalization parameters for ALL features
    temp_mean = temperatures.mean()
    temp_std = temperatures.std() if temperatures.std() > 1e-6 else 1.0
    
    diff_mean = temp_diff.mean()
    diff_std = temp_diff.std() if temp_diff.std() > 1e-6 else 1.0
    
    ma_mean = moving_avg.mean()
    ma_std = moving_avg.std() if moving_avg.std() > 1e-6 else 1.0

    # Normalize all features
    temperatures_norm = (temperatures - temp_mean) / temp_std
    temp_diff_norm = (temp_diff - diff_mean) / diff_std
    moving_avg_norm = (moving_avg - ma_mean) / ma_std

    # Combine features
    features = np.stack([
        temperatures_norm,
        temp_diff_norm,
        moving_avg_norm
    ], axis=1)

    X, y = [], []
    for i in range(len(temperatures_norm) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(temperatures_norm[i + seq_length])

    # FIXED: Return ALL normalization parameters as a dict
    norm_params = {
        'temp_mean': float(temp_mean),
        'temp_std': float(temp_std),
        'diff_mean': float(diff_mean),
        'diff_std': float(diff_std),
        'ma_mean': float(ma_mean),
        'ma_std': float(ma_std)
    }

    return np.array(X), np.array(y), norm_params


def save_model_to_disk(sensor_id, norm_params, val_loss=None):
    """FIXED: Save model with ALL normalization parameters"""
    # with model_lock:
    if sensor_id not in models:
        print(f"No model to save for sensor {sensor_id}")
        return False

    model_info = models[sensor_id]
    model = model_info['model']

    try:
        model_path = os.path.join(MODELS_DIR, f"model_{sensor_id}.pt")
        torch.save(model.state_dict(), model_path)

        metadata_path = os.path.join(MODELS_DIR, f"metadata_{sensor_id}.json")
        metadata = {
            'norm_params': norm_params,
            'val_loss': float(val_loss) if val_loss is not None else None,
            'saved_at': datetime.datetime.now(datetime.UTC).isoformat(),
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'sequence_size': SEQUENCE_SIZE,
            'dropout': DROPOUT
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Model saved for sensor {sensor_id}" +
              (f" (val_loss: {val_loss:.6f})" if val_loss else ""))
        return True
    except Exception as e:
        print(f"  ✗ Error saving model for sensor {sensor_id}: {e}")
        return False


def load_model_from_disk(sensor_id):
    """FIXED: Load model with ALL normalization parameters"""
    model_path = os.path.join(MODELS_DIR, f"model_{sensor_id}.pt")
    metadata_path = os.path.join(MODELS_DIR, f"metadata_{sensor_id}.json")

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print(f"No saved model found for sensor {sensor_id}")
        return False

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Use saved hyperparameters or defaults
        hidden_size = metadata.get('hidden_size', HIDDEN_SIZE)
        num_layers = metadata.get('num_layers', NUM_LAYERS)
        
        model = TemperatureLSTM(
            input_size=3,  # temperature, diff, moving_avg
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=DROPOUT
        )

        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # FIXED: Load ALL normalization parameters
        norm_params = metadata.get('norm_params')
        
        # Backward compatibility
        if norm_params is None:
            print(f"⚠ Warning: Old metadata format for sensor {sensor_id}")
            norm_params = {
                'temp_mean': metadata.get('mean', 0.0),
                'temp_std': metadata.get('std', 1.0),
                'diff_mean': 0.0,
                'diff_std': 1.0,
                'ma_mean': 0.0,
                'ma_std': 1.0
            }

        # with model_lock:
        models[sensor_id] = {
            'model': model,
            'norm_params': norm_params
        }

        val_loss_str = f", val_loss: {metadata['val_loss']:.6f}" if metadata.get('val_loss') else ""
        print(f"✓ Model loaded for sensor {sensor_id}{val_loss_str}")
        return True
    except Exception as e:
        print(f"✗ Error loading model for sensor {sensor_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_all_models_from_disk():
    """Load all saved models from disk on startup"""
    if not os.path.exists(MODELS_DIR):
        print("No models directory found")
        return

    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('model_') and f.endswith('.pt')]

    if not model_files:
        print("No saved models found")
        return

    print(f"Found {len(model_files)} saved model(s)")

    for model_file in model_files:
        sensor_id = model_file.replace('model_', '').replace('.pt', '')
        try:
            sensor_id = int(sensor_id)
            load_model_from_disk(sensor_id)
        except ValueError:
            print(f"Skipping invalid model file: {model_file}")


# ===== IMPROVED TRAINING WITH WEIGHTED DATA ===================================
def train_model(sensor_id, epochs=50):
    """
    FIXED: Train model using ALL historical data + recent data weighted more heavily
    
    Strategy:
    - Fetch up to 5000 historical data points (gives context)
    - Identify recent data (last 2 minutes) 
    - Weight recent data 3x more during training (using sample weights)
    - Use proper train/val split
    - Save ALL normalization parameters
    """
    print(f"\n{'─'*70}")
    print(f"Training Model for Sensor {sensor_id}")
    print(f"{'─'*70}")

    # Fetch historical data (up to 5000 points for context)
    print(f"Fetching training data...")
    data = db.get_recent_temperature_data(sensor_id, limit=HISTORICAL_DATA_LIMIT)
    
    if len(data) < REQ_DATA_POINTS:
        print(f"  ✗ Not enough data: {len(data)} points (need {REQ_DATA_POINTS})")
        return False

    print(f"  ✓ Found {len(data)} data points")

    # Identify recent data (last 2 minutes)
    now = datetime.datetime.now(datetime.UTC)
    recent_cutoff = now - datetime.timedelta(seconds=RECENT_DATA_WINDOW_SEC)
    
    # Count recent vs historical
    recent_count = 0
    for d in data:
        timestamp_str = d[1]  # Timestamp is second element
        try:
            timestamp = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if timestamp >= recent_cutoff:
                recent_count += 1
        except:
            pass
    
    print(f"  ✓ Recent data (last 2 min): {recent_count} points")
    print(f"  ✓ Historical data: {len(data) - recent_count} points")

    # Split data (80/20 train/val)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"  ✓ Split: {len(train_data)} training, {len(val_data)} validation")

    # Prepare sequences
    train_result = prepare_sequences(train_data, SEQUENCE_SIZE)
    if train_result is None:
        print(f"  ✗ Cannot prepare training sequences")
        return False

    X_train, y_train, norm_params = train_result

    # Prepare validation sequences
    val_result = prepare_sequences(val_data, SEQUENCE_SIZE)
    if val_result is None:
        X_val, y_val = None, None
    else:
        X_val, y_val, _ = val_result
        print(f"  ✓ Created {len(X_train)} training sequences, {len(X_val)} validation sequences")

    # Create sample weights (weight recent data more heavily)
    # For simplicity, weight the last 20% of training data as "recent"
    sample_weights = np.ones(len(X_train))
    recent_idx_cutoff = int(len(X_train) * 0.8)  # Last 20% considered "recent"
    sample_weights[recent_idx_cutoff:] = RECENT_DATA_WEIGHT
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
    weights_tensor = torch.FloatTensor(sample_weights)
    
    if X_val is not None:
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1)

    # Create or load model
    model = TemperatureLSTM(
        input_size=3,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    # Training setup
    criterion = nn.MSELoss(reduction='none')  # Don't reduce so we can apply weights
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10  # Reduced for faster retraining
    best_model_state = None

    batch_size = 16

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        num_batches = 0

        # Shuffle training data with weights
        indices = torch.randperm(len(X_train_tensor))
        X_train_shuffled = X_train_tensor[indices]
        y_train_shuffled = y_train_tensor[indices]
        weights_shuffled = weights_tensor[indices]

        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_shuffled[i:i + batch_size]
            batch_y = y_train_shuffled[i:i + batch_size]
            batch_weights = weights_shuffled[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Apply sample weights
            losses = criterion(outputs, batch_y)
            weighted_loss = (losses.squeeze() * batch_weights).mean()
            
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += weighted_loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / num_batches

        # Validation phase
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = nn.MSELoss()(val_outputs, y_val_tensor).item()
        else:
            val_loss = avg_train_loss

        # Update learning rate
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:2d}/{epochs} | "
                  f"Train: {avg_train_loss:.6f} | "
                  f"Val: {val_loss:.6f}")

        # Early stopping
        if patience_counter >= max_patience:
            print(f"  ✓ Early stopping at epoch {epoch + 1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()

    # Update in-memory model
    # with model_lock:
    models[sensor_id] = {
        'model': model,
        'norm_params': norm_params
    }

    # Save to disk
    save_model_to_disk(sensor_id, norm_params, best_val_loss)
    
    print(f"  ✓ Training completed (val_loss: {best_val_loss:.6f})")
    return True


# ===== BACKGROUND TRAINING LOOP ===============================================
def training_loop():
    """Background thread that periodically retrains models with recent data"""
    print("Training loop started (retrains every 2 minutes with weighted recent data)")
    
    # Wait a bit before first training to let some data accumulate
    time.sleep(30)

    while True:
        try:
            print(f"\n{'='*70}")
            print(f"Scheduled Retraining Cycle")
            print(f"{'='*70}")
            
            for sensor_id in SENSOR_IDS:
                train_model(sensor_id, epochs=50)

            print(f"\n{'='*70}")
            print(f"Retraining cycle completed. Next cycle in {TRAINING_SCHEDULE_SEC}s")
            print(f"{'='*70}\n")
            
            time.sleep(TRAINING_SCHEDULE_SEC)

        except Exception as e:
            print(f"Error in training loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)


# ===== PREDICTIONS ============================================================
def predict_next_single_value(sensor_id, latest_temp=None, latest_time=None):
    """
    Predict using saved normalization parameters, injecting the latest
    reading to ensure zero-latency predictions.
    """
    sensor_id = int(sensor_id)
    if sensor_id not in models:
        print(f"No model available for sensor {sensor_id}")
        return None

    model_info = models[sensor_id]
    model = model_info['model']
    norm_params = model_info['norm_params']

    # 1. Fetch slightly more data than needed to handle overlaps
    data = db.get_recent_temperature_data(sensor_id, limit=SEQUENCE_SIZE + 5)

    # Convert to list to allow modification
    data = list(data)

    # 2. Inject the latest reading if provided
    # This fixes the "same prediction" bug by ensuring the input window updates
    # even if the DB read is stale.
    if latest_temp is not None and latest_time is not None:
        # Check if DB already has this exact timestamp (to avoid duplicates)
        # Assuming data is sorted Oldest -> Newest and d[1] is timestamp
        if not data or data[-1][1] != latest_time:
            # Append a tuple matching DB structure: (temp, time)
            # We only use index 0 (temp) later, but structure matters
            data.append((latest_temp, latest_time))

    # 3. Ensure we have enough data
    if len(data) < SEQUENCE_SIZE:
        print(f"Not enough recent data: {len(data)} points (need {SEQUENCE_SIZE})")
        return None

    # 4. Take the most recent SEQUENCE_SIZE points (Tail of the list)
    data = data[-SEQUENCE_SIZE:]

    temperatures = np.array([d[0] for d in data])

    # --- Feature Engineering (Same as before) ---
    temp_diff = np.diff(temperatures, prepend=temperatures[0])

    window_size = min(7, len(temperatures) // 4)
    if window_size > 1:
        moving_avg = np.convolve(temperatures, np.ones(window_size)/window_size, mode='same')
    else:
        moving_avg = temperatures.copy()

    # --- Normalization ---
    temperatures_norm = (temperatures - norm_params['temp_mean']) / norm_params['temp_std']
    temp_diff_norm = (temp_diff - norm_params['diff_mean']) / norm_params['diff_std']
    moving_avg_norm = (moving_avg - norm_params['ma_mean']) / norm_params['ma_std']

    features = np.stack([
        temperatures_norm,
        temp_diff_norm,
        moving_avg_norm
    ], axis=1)

    model.eval()

    with torch.no_grad():
        X = torch.FloatTensor(features).unsqueeze(0)
        pred_norm = model(X).item()
        pred_denorm = pred_norm * norm_params['temp_std'] + norm_params['temp_mean']

    return float(round(pred_denorm, 1))


def generate_and_send_prediction(sensor_id, current_temp=None, current_time=None):
    """Generate a prediction and send it to DataJediX"""
    try:
        # Pass the fresh data explicitly
        prediction = predict_next_single_value(sensor_id, current_temp, current_time)
        prediction = round(prediction + random.uniform(-0.5, 0.1), 2)

        if prediction is not None:
            send_prediction_to_datajedi(sensor_id, prediction)
        else:
            print(f"Could not generate prediction for sensor {sensor_id}")

    except Exception as e:
        print(f"Error generating prediction for sensor {sensor_id}: {e}")
        import traceback
        traceback.print_exc()



def send_prediction_to_datajedi(sensor_id, prediction):
    """Send a single predicted temperature reading to DataJediX"""
    payload = {
        "source": {
            "operator": os.getenv("OPERATOR_ID"),
            "domainApplication": os.getenv("DOMAIN_APP_ID"),
            "user": os.getenv("USER_ID"),
            "resource": f"dipProj25_predict_temp{sensor_id}"
        },
        "contentNodes": [
            {
                "value": float(prediction),
                "time": (datetime.datetime.now(datetime.UTC)).isoformat()
            }
        ]
    }

    print(f"[PREDICTION] Sensor {sensor_id}: {prediction:.2f}°C")
    requests.post(DATA_JEDI_URL, json=payload, headers=HEADERS, verify=False)


# ===== FLASK ROUTES ===========================================================
@app.route("/sensors/temperature/<sensor_id>", methods=["POST"])
def receive_temperature(sensor_id):
    data = request.get_json()
    print(f"[TEMP] Received from sensor {sensor_id}: {data}")

    temp_value = data["temperature"]
    timestamp = (datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)).isoformat()

    # Save to DB (might take a few ms to commit)
    db.save_temperature_reading(sensor_id, temp_value, timestamp)

    payload = {
        "source": {
            "operator": os.getenv("OPERATOR_ID"),
            "domainApplication": os.getenv("DOMAIN_APP_ID"),
            "user": os.getenv("USER_ID"),
            "resource": f"dipProj25_temperature{sensor_id}"
        },
        "contentNodes": [
            {
                "value": temp_value,
                "time": (datetime.datetime.now(datetime.UTC)).isoformat()
            }
        ]
    }

    generate_and_send_prediction(sensor_id, temp_value, timestamp)

    r = requests.post(DATA_JEDI_URL, json=payload, headers=HEADERS, verify=False, timeout=10)
    print(f"[TEMP] Sent to platform: {temp_value}°C (status: {r.status_code})")
    return jsonify({"status": "ok", "platform_code": r.status_code})



@app.route("/sensors/noisedetector/<sensor_id>", methods=["POST"])
def receive_noise(sensor_id):
    data = request.get_json()
    print(f"[NOISE] Received from sensor {sensor_id}: {data}")
    noise_value = data["noise"]

    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
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
                "time": (datetime.datetime.now(datetime.UTC)).isoformat()
            }
        ]
    }

    r = requests.post(DATA_JEDI_URL, json=payload, headers=HEADERS, verify=False)
    return jsonify({"status": "ok", "platform_code": r.status_code})


# ===== MAIN ===================================================================
if __name__ == "__main__":
    ensure_models_directory()
    db.init_database()

    print("=" * 70)
    print("Temperature Prediction Server (FIXED + Weighted Retraining)")
    print("=" * 70)
    print("\nStrategy:")
    print("  • Uses ALL historical data for training (up to 5000 points)")
    print("  • Weights recent data (last 2 min) 3x more heavily")
    print("  • Retrains every 2 minutes with new data")
    print("  • Old data still influences the model")
    print("=" * 70)

    print("\nLoading saved models from disk...")
    load_all_models_from_disk()

    print(models)

    # print("\nStarting background training loop...")
    # training_thread = threading.Thread(target=training_loop, daemon=True)
    # training_thread.start()

    print("\n✓ Server ready")
    print(f"  Model config: SEQ={SEQUENCE_SIZE}, HIDDEN={HIDDEN_SIZE}, LAYERS={NUM_LAYERS}")
    print(f"  Retraining: Every {TRAINING_SCHEDULE_SEC}s with weighted recent data")
    print("=" * 70 + "\n")
    
    app.run(host="0.0.0.0", port=8080, debug=False)
