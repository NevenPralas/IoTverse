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
import json
import db
import urllib3


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
INITIAL_TRAINING_DELAY_SEC = 20
TRAINING_SCHEDULE_SEC = 120

REQ_DATA_POINTS = 200
SEQUENCE_SIZE = 60
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.2


# ===== CONFIG =================================================================
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
app = Flask(__name__)


# ===== GLOBALS ================================================================
models = {}
model_lock = threading.Lock()


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
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        x = self.dropout(last_output)
        
        # First FC layer with activation
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Final prediction
        prediction = self.fc2(x)
        return prediction


def ensure_models_directory():
    """Ensure the models directory exists"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"✓ Created models directory: {MODELS_DIR}")


def prepare_sequences(data, seq_length=60):
    """Prepare sequences for training with enhanced features"""
    if len(data) < seq_length + 1:
        return None

    temperatures = np.array([d[0] for d in data])

    # Calculate rate of change (temperature derivative)
    temp_diff = np.diff(temperatures, prepend=temperatures[0])
    
    # Calculate moving averages for trend detection
    window_size = min(7, len(temperatures) // 4)
    if window_size > 1:
        moving_avg = np.convolve(temperatures, np.ones(window_size)/window_size, mode='same')
    else:
        moving_avg = temperatures.copy()

    # Normalize data with epsilon to avoid division by zero
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

    # Combine features: temperature, rate of change, and moving average
    features = np.stack([
        temperatures_norm,
        temp_diff_norm,
        moving_avg_norm
    ], axis=1)

    X, y = [], []
    for i in range(len(temperatures_norm) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(temperatures_norm[i + seq_length])

    return np.array(X), np.array(y), temp_mean, temp_std


def save_model_to_disk(sensor_id, val_loss=None):
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
        model_path = os.path.join(MODELS_DIR, f"model_{sensor_id}.pt")
        torch.save(model.state_dict(), model_path)

        metadata_path = os.path.join(MODELS_DIR, f"metadata_{sensor_id}.json")
        metadata = {
            'mean': float(mean),
            'std': float(std),
            'val_loss': float(val_loss) if val_loss is not None else None,
            'saved_at': datetime.datetime.now(datetime.UTC).isoformat(),
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'sequence_size': SEQUENCE_SIZE
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to disk for sensor {sensor_id}" +
              (f" (val_loss: {val_loss:.6f})" if val_loss else ""))
        return True
    except Exception as e:
        print(f"Error saving model for sensor {sensor_id}: {e}")
        return False


def load_model_from_disk(sensor_id):
    """Load the model and its metadata from disk"""
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

        with model_lock:
            models[sensor_id] = {
                'model': model,
                'mean': metadata['mean'],
                'std': metadata['std']
            }

        val_loss_str = f", val_loss: {metadata['val_loss']:.6f}" if metadata.get('val_loss') else ""
        print(f"Model loaded from disk for sensor {sensor_id} (saved at {metadata.get('saved_at', 'unknown')}{val_loss_str})")
        return True
    except Exception as e:
        print(f"Error loading model for sensor {sensor_id}: {e}")
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
        load_model_from_disk(sensor_id)


def train_model(sensor_id):
    """
    Train or update the model for a specific sensor with improvements:
    - Train/validation split
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Best model checkpointing
    """
    print(f"Training model for sensor: {sensor_id}")

    # Fetch more data for better training (500 points if available)
    data = db.get_recent_temperature_data(sensor_id, limit=500)

    if len(data) < REQ_DATA_POINTS:
        print(f"Not enough data for sensor {sensor_id}: {len(data)} points (need {REQ_DATA_POINTS})")
        return

    # Split data into train and validation (80/20 split)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Prepare sequences for training
    train_result = prepare_sequences(train_data, SEQUENCE_SIZE)
    if train_result is None:
        print(f"Cannot prepare training sequences for sensor {sensor_id}")
        return

    X_train, y_train, mean, std = train_result

    # Prepare sequences for validation
    val_result = prepare_sequences(val_data, SEQUENCE_SIZE)
    if val_result is None:
        print(f"Not enough validation data, using training data only")
        X_val, y_val = None, None
    else:
        X_val, y_val, _, _ = val_result

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
    
    if X_val is not None:
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1)

    # Create or update model
    with model_lock:
        if sensor_id not in models:
            models[sensor_id] = {
                'model': TemperatureLSTM(
                    input_size=3,  # temperature, diff, moving_avg
                    hidden_size=HIDDEN_SIZE,
                    num_layers=NUM_LAYERS,
                    dropout=DROPOUT
                ),
                'mean': mean,
                'std': std
            }
        else:
            # Update normalization parameters
            models[sensor_id]['mean'] = mean
            models[sensor_id]['std'] = std

        model = models[sensor_id]['model']

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15
    best_model_state = None

    model.train()
    epochs = 100
    batch_size = 16

    print(f"Training with {len(X_train)} samples, validating with {len(X_val) if X_val is not None else 0} samples")

    for epoch in range(epochs):
        # === TRAINING PHASE ===
        model.train()
        total_train_loss = 0
        num_batches = 0

        # Shuffle training data
        indices = torch.randperm(len(X_train_tensor))
        X_train_shuffled = X_train_tensor[indices]
        y_train_shuffled = y_train_tensor[indices]

        # Mini-batch training
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_shuffled[i:i + batch_size]
            batch_y = y_train_shuffled[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / num_batches

        # === VALIDATION PHASE ===
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
        else:
            val_loss = avg_train_loss

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Sensor {sensor_id} - Epoch [{epoch + 1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"LR: {current_lr:.6f}, Patience: {patience_counter}/{max_patience}")

        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch + 1} (best val_loss: {best_val_loss:.6f})")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with val_loss: {best_val_loss:.6f}")

    model.eval()
    print(f"Model training completed for sensor: {sensor_id}")
    save_model_to_disk(sensor_id, best_val_loss)


# ===== BACKGROUND TRAINING LOOP ===============================================
def training_loop():
    """Background thread that periodically trains models"""
    print("Training loop started")

    while True:
        try:
            time.sleep(TRAINING_SCHEDULE_SEC)
            sensor_ids = SENSOR_IDS
            for sensor_id in sensor_ids:
                print(f"Starting training for sensor {sensor_id}...")
                train_model(sensor_id)
                print(f"Training completed for sensor {sensor_id}")

            print(f"Training cycle completed. Next training in {TRAINING_SCHEDULE_SEC} seconds...")

        except Exception as e:
            print(f"Error in training loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)


# ===== PREDICTIONS ============================================================
def predict_next_single_value(sensor_id):
    """
    Predict only the next single value using the data from the database.
    
    Improvements:
    - Uses enhanced features (temperature, diff, moving average)
    - Matches the sequence size used during training
    """
    with model_lock:
        if sensor_id not in models:
            print(f"No model available for sensor {sensor_id}")
            return None

        model_info = models[sensor_id]
        model = model_info['model']
        mean = model_info['mean']
        std = model_info['std']

    # Fetch data matching the sequence size
    data = db.get_recent_temperature_data(sensor_id, limit=SEQUENCE_SIZE)

    if len(data) < SEQUENCE_SIZE:
        print(f"Not enough recent data for prediction: {len(data)} points (need {SEQUENCE_SIZE})")
        return None

    temperatures = np.array([d[0] for d in data])
    
    # Calculate features (same as training)
    temp_diff = np.diff(temperatures, prepend=temperatures[0])
    
    window_size = min(7, len(temperatures) // 4)
    if window_size > 1:
        moving_avg = np.convolve(temperatures, np.ones(window_size)/window_size, mode='same')
    else:
        moving_avg = temperatures.copy()
    
    # Normalize using training statistics
    temperatures_norm = (temperatures - mean) / std
    
    # For diff and moving_avg, use their own normalization
    diff_mean = temp_diff.mean()
    diff_std = temp_diff.std() if temp_diff.std() > 1e-6 else 1.0
    temp_diff_norm = (temp_diff - diff_mean) / diff_std
    
    ma_mean = moving_avg.mean()
    ma_std = moving_avg.std() if moving_avg.std() > 1e-6 else 1.0
    moving_avg_norm = (moving_avg - ma_mean) / ma_std
    
    # Combine features
    features = np.stack([
        temperatures_norm,
        temp_diff_norm,
        moving_avg_norm
    ], axis=1)

    model.eval()

    with torch.no_grad():
        X = torch.FloatTensor(features).unsqueeze(0)  # Shape: (1, seq_len, 3)
        pred_norm = model(X).item()
        pred_denorm = pred_norm * std + mean
    
    return pred_denorm


def generate_and_send_prediction(sensor_id):
    """Generate a prediction and send it to DataJediX when temperature reading is received"""
    try:
        with model_lock:
            if sensor_id not in models:
                print(f"No model available for sensor {sensor_id}, skipping prediction")
                return

        prediction = predict_next_single_value(sensor_id)

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
    if prediction is None:
        print(f"No prediction to send for sensor {sensor_id}")
        return

    future_time = datetime.datetime.now(datetime.UTC).replace(microsecond=0) + datetime.timedelta(seconds=30)

    payload = {
        "source": {
            "operator": os.getenv("OPERATOR_ID"),
            "domainApplication": os.getenv("DOMAIN_APP_ID"),
            "user": os.getenv("USER_ID"),
            "res": f"dipProj25_temperature_prediction{sensor_id}"
        },
        "contentNodes": [
            {
                "value": float(prediction),
                "time": future_time.isoformat()
            }
        ]
    }

    try:
        r = requests.post(DATA_JEDI_URL, json=payload, headers=HEADERS, verify=False)
        print(f"Sent prediction for sensor {sensor_id}: {prediction:.2f}°C (status: {r.status_code})")
    except Exception as e:
        print(f"Error sending prediction for sensor {sensor_id}: {e}")


# ===== FLASK ROUTES ===========================================================
@app.route("/sensors/temperature/<sensor_id>", methods=["POST"])
def receive_temperature(sensor_id):
    data = request.get_json()
    print(f"Received temperature from sensor {sensor_id}: {data}")
    temp_value = data.get("temperature")

    if temp_value is None:
        return jsonify({"error": "Missing temperature value"}), 400

    timestamp = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()
    db.save_temperature_reading(sensor_id, temp_value, timestamp)

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
    generate_and_send_prediction(sensor_id)
    return jsonify({"status": "ok", "platform_code": r.status_code})


@app.route("/sensors/noisedetector/<sensor_id>", methods=["POST"])
def receive_noise(sensor_id):
    data = request.get_json()
    print(f"Received noise from {sensor_id}: {data}")
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


# ===== MAIN ===================================================================
if __name__ == "__main__":
    ensure_models_directory()
    db.init_database()

    print("Loading saved models from disk...")
    load_all_models_from_disk()

    training_thread = threading.Thread(target=training_loop, daemon=True)
    training_thread.start()

    print("Server ready - predictions will be generated on-demand when temperature readings are received")
    print(f"Model config: SEQUENCE_SIZE={SEQUENCE_SIZE}, HIDDEN_SIZE={HIDDEN_SIZE}, NUM_LAYERS={NUM_LAYERS}")
    app.run(host="0.0.0.0", port=8080, debug=False)
