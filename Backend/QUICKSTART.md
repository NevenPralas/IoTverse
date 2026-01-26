# Quick Start Guide

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file with your DataJediX credentials:
```bash
cp .env.example .env
# Edit .env and add your credentials
```

### 3. Start the Server
```bash
python app.py
```

The server will:
- Initialize the SQLite database
- Start the Flask API on port 8080
- Launch background training thread (trains every 2 minutes)
- Launch background prediction thread (predicts every 30 seconds)

## Testing the System

### Option 1: Automated Test (Recommended)
```bash
python test_system.py
```
Choose option 5 for a full test that will:
1. Generate 5 minutes of realistic sensor data
2. Wait for models to train
3. Test predictions
4. Show statistics

### Option 2: Manual Testing
Send test data manually:
```bash
curl -X POST http://localhost:8080/sensors/temperature \
  -H "Content-Type: application/json" \
  -d '{"temperature": {"sensor1": 25.5, "sensor2": 26.0, "sensor3": 24.8, "sensor4": 25.2}}'
```

Check statistics:
```bash
curl http://localhost:8080/api/stats
```

Get predictions:
```bash
curl http://localhost:8080/api/predict/sensor1
```

## Monitoring

### View Database Contents
```bash
python monitor_db.py
```

### Check Logs
The application prints logs to stdout:
- Temperature readings received
- Model training progress
- Predictions sent to DataJediX
- Errors and warnings

## Expected Behavior

### Initial Phase (First 2 minutes)
- Server receives temperature data
- Data is saved to database
- Not enough data for training yet

### Training Phase (After 2-3 minutes)
- Training loop detects sufficient data (60+ readings)
- Models start training (check logs for "Training model for sensor: ...")
- Training takes ~2-5 seconds per sensor

### Prediction Phase (After 3-4 minutes)
- Models are trained and ready
- Prediction loop generates forecasts every 30 seconds
- Predictions are sent to DataJediX
- Frontend can now display predictions

## Timeline Summary

| Time | What Happens |
|------|-------------|
| 0:00 | Server starts, database initialized |
| 0:01 | First sensor readings arrive |
| 0:30 | Training loop first check (not enough data) |
| 1:00 | Prediction loop first check (no models yet) |
| 2:00 | Training loop starts (60+ readings available) |
| 2:05 | First models trained successfully |
| 2:30 | First predictions generated and sent |
| 3:00+ | System running normally |

## Docker Deployment (Optional)

### Build and Run
```bash
docker-compose up -d
```

### View Logs
```bash
docker-compose logs -f
```

### Stop
```bash
docker-compose down
```

## Common Issues

### "No model available for sensor"
- **Cause**: Not enough data yet or training hasn't run
- **Solution**: Wait for at least 60 readings (60 seconds of data)

### "Not enough recent data for prediction"
- **Cause**: No data in last 2 minutes
- **Solution**: Ensure sensors are actively sending data

### Port 8080 already in use
- **Solution**: Change port in `app.py` or stop other service using port 8080

### Database locked errors
- **Cause**: Multiple simultaneous database access (rare)
- **Solution**: Application handles this automatically, errors should be transient

## Verification Checklist

✅ Server is running (check `http://localhost:8080/api/stats`)
✅ Database file `sensors.db` exists
✅ Temperature readings are being received (check logs)
✅ Models are training (check logs for "Model training completed")
✅ Predictions are being generated (check logs for "Prediction sent")
✅ DataJediX is receiving data (check platform)

## Next Steps

1. **Integrate with Real Sensors**: Replace test script with actual sensor data
2. **Monitor Performance**: Use `monitor_db.py` to track data quality
3. **Tune Models**: Adjust hyperparameters in `app.py` if needed
4. **Scale Up**: Add more sensors by sending data with different `sensor_id` values

## Support

For detailed information, see `README.md`

For issues:
1. Check server logs for error messages
2. Verify database has data (`/api/stats`)
3. Ensure .env credentials are correct
4. Check network connectivity to DataJediX
