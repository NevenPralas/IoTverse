# Temperature Sensor Prediction System

A Flask-based IoT backend that receives temperature sensor readings, stores them in SQLite, trains PyTorch LSTM models, and predicts future temperatures.

## Features

- **Data Storage**: SQLite database for persistent storage of temperature and noise sensor readings
- **Machine Learning**: PyTorch LSTM model for time series prediction
- **Automatic Training**: Background thread that trains models every 2 minutes
- **Automatic Predictions**: Background thread that generates and sends predictions every 30 seconds
- **Multi-Sensor Support**: Handles multiple temperature sensors independently
- **DataJediX Integration**: Sends both actual and predicted readings to external IoT platform

## Architecture

### Components

1. **Flask API**: Receives sensor data via HTTP POST
2. **SQLite Database**: Stores historical sensor readings
3. **PyTorch LSTM Model**: Predicts next 30 seconds of temperature readings
4. **Background Threads**:
   - Training loop: Retrains models every 2 minutes
   - Prediction loop: Generates predictions every 30 seconds

### Data Flow

```
Sensor → POST /sensors/temperature → Database → Training Thread → Model
                                                                    ↓
Frontend ← DataJediX ← Prediction Thread ← Model
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your credentials:
```env
OPERATOR_ID=your_operator_id
DOMAIN_APP_ID=your_domain_app_id
USER_ID=your_user_id
```

3. Run the application:
```bash
python app.py
```

## API Endpoints

### Receive Temperature Data
```http
POST /sensors/temperature
Content-Type: application/json

{
  "sensor_id": "sensor1",
  "temperature": 25.5
}
```

Or for multiple sensors:
```http
POST /sensors/temperature
Content-Type: application/json

{
  "temperature": {
    "sensor1": 25.5,
    "sensor2": 26.0,
    "sensor3": 24.8,
    "sensor4": 25.2
  }
}
```

### Receive Noise Data
```http
POST /sensors/noisedetector
Content-Type: application/json

{
  "noise": 45.2
}
```

### Get Prediction (Manual)
```http
GET /api/predict/<sensor_id>
```

Response:
```json
{
  "sensor_id": "sensor1",
  "predictions": [25.5, 25.6, 25.7, ...]
}
```

### Trigger Training (Manual)
```http
POST /api/train/<sensor_id>
```

### Get Statistics
```http
GET /api/stats
```

Response:
```json
{
  "temperature_readings": {
    "sensor1": 1250,
    "sensor2": 1248,
    "sensor3": 1251,
    "sensor4": 1249
  },
  "noise_readings": 1250,
  "active_models": ["sensor1", "sensor2", "sensor3", "sensor4"]
}
```

## Model Details

### LSTM Architecture
- **Input**: Sequence of 30 temperature readings
- **Hidden Layers**: 2 LSTM layers with 64 hidden units each
- **Output**: Single temperature prediction
- **Prediction Method**: Autoregressive (uses previous predictions for future predictions)

### Training Process
- **Sequence Length**: 30 readings (30 seconds)
- **Training Data**: Last 30 minutes of readings
- **Normalization**: Z-score normalization (mean/std)
- **Epochs**: 50
- **Batch Size**: 16
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: MSE (Mean Squared Error)

### Prediction Process
1. Retrieves last 30 readings from database
2. Normalizes the data
3. Predicts next reading
4. Uses prediction as input for next prediction (autoregressive)
5. Repeats for 30 predictions (30 seconds)
6. Denormalizes predictions
7. Sends to DataJediX with future timestamps

## Database Schema

### temperature_readings
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| sensor_id | TEXT | Sensor identifier |
| temperature | REAL | Temperature value |
| timestamp | DATETIME | Reading timestamp |
| created_at | DATETIME | Record creation time |

### noise_readings
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| noise | REAL | Noise level |
| timestamp | DATETIME | Reading timestamp |
| created_at | DATETIME | Record creation time |

## Background Processes

### Training Loop
- **Frequency**: Every 2 minutes
- **Initial Delay**: 30 seconds (to collect initial data)
- **Action**: Trains model for each sensor with sufficient data
- **Minimum Data**: 60 readings required

### Prediction Loop
- **Frequency**: Every 30 seconds
- **Initial Delay**: 60 seconds (to allow initial training)
- **Action**: Generates 30-second predictions and sends to DataJediX
- **Format**: Predictions sent as `temperature_prediction_{sensor_id}`

## DataJediX Integration

### Actual Readings
Sent to DataJediX with:
- **Resource Spec**: `temperature`
- **Content**: Current temperature value
- **Timestamp**: Current UTC time

### Predicted Readings
Sent to DataJediX with:
- **Resource Spec**: `temperature_prediction_{sensor_id}`
- **Content**: Predicted temperature value
- **Timestamp**: Future UTC time (+1 to +30 seconds)

## Usage Example

### Sending Data from Sensors

```python
import requests
import time

# Simulate four temperature sensors
while True:
    data = {
        "temperature": {
            "sensor1": 25.5 + random.random(),
            "sensor2": 26.0 + random.random(),
            "sensor3": 24.8 + random.random(),
            "sensor4": 25.2 + random.random()
        }
    }
    
    response = requests.post(
        "http://localhost:8080/sensors/temperature",
        json=data
    )
    print(f"Response: {response.status_code}")
    
    time.sleep(1)  # Send every second
```

### Checking Model Performance

```python
import requests

# Get statistics
stats = requests.get("http://localhost:8080/api/stats").json()
print(f"Active models: {stats['active_models']}")
print(f"Data points: {stats['temperature_readings']}")

# Get prediction for sensor1
prediction = requests.get("http://localhost:8080/api/predict/sensor1").json()
print(f"Next 30 seconds: {prediction['predictions']}")
```

## Frontend Integration

The frontend expects:
- **Historical Data**: Last 30 seconds from DataJediX (resourceSpec: `temperature`)
- **Predicted Data**: Next 30 seconds from DataJediX (resourceSpec: `temperature_prediction_{sensor_id}`)
- **Update Frequency**: Graph updates every second

## Troubleshooting

### Model Not Training
- **Cause**: Not enough data points
- **Solution**: Wait for at least 60 readings per sensor (60 seconds of data)

### No Predictions Available
- **Cause**: Model not trained yet or insufficient recent data
- **Solution**: Wait for training loop to complete and ensure sensors are sending data

### Database Locked Errors
- **Cause**: Multiple threads accessing database simultaneously
- **Solution**: The app handles this automatically with proper connection management

### Poor Prediction Accuracy
- **Cause**: Insufficient training data or highly irregular patterns
- **Solution**: 
  - Collect more data (models improve with more training data)
  - Adjust training parameters (epochs, learning rate)
  - Increase training frequency

## Configuration

### Model Hyperparameters
Edit in `app.py`:
```python
# LSTM architecture
hidden_size = 64  # Number of LSTM hidden units
num_layers = 2    # Number of LSTM layers

# Training
epochs = 50       # Training epochs
batch_size = 16   # Batch size
learning_rate = 0.001  # Adam learning rate
seq_length = 30   # Input sequence length
```

### Background Thread Timing
```python
# Training loop
time.sleep(120)  # Train every 2 minutes

# Prediction loop
time.sleep(30)   # Predict every 30 seconds
```

### Data Retention
```python
# In get_recent_temperature_data()
minutes=10  # Keep last 10 minutes for training
```

## Performance Notes

- **Training Time**: ~2-5 seconds per sensor (depends on data volume)
- **Prediction Time**: ~100ms per sensor
- **Memory Usage**: ~50MB + (10MB per active model)
- **Database Size**: ~1KB per 100 readings

## License

MIT

## Support

For issues or questions, please check:
1. Database has sufficient data (`/api/stats`)
2. Models are trained (check logs)
3. DataJediX credentials are correct (.env file)
4. Network connectivity to DataJediX platform
