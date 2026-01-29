# Training Guide - Temperature Prediction Models

## Quick Start

### Option 1: Train All Sensors (Recommended)
```bash
python train_models.py
```

### Option 2: Train Specific Sensor
```bash
python train_models.py --sensor 1
```

### Option 3: Train Multiple Sensors
```bash
python train_models.py --sensor 1 2 3
```

## Common Usage Examples

### Detailed Progress Output
```bash
python train_models.py --verbose
```
Shows loss for every epoch instead of every 10 epochs.

### Custom Training Duration
```bash
python train_models.py --epochs 200
```
Train for up to 200 epochs (default is 100, but early stopping may stop sooner).

### Lower Data Requirements
```bash
python train_models.py --min-data 100
```
Train with minimum 100 data points instead of 200 (useful when starting fresh).

### Combined Options
```bash
python train_models.py --sensor 1 2 --epochs 150 --verbose
```

## Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--sensor` | `-s` | Which sensor(s) to train | All (1,2,3,4) |
| `--epochs` | `-e` | Maximum training epochs | 100 |
| `--verbose` | `-v` | Show detailed progress | False |
| `--min-data` | | Minimum data points needed | 200 |
| `--help` | `-h` | Show help message | |

## Training Workflow

### 1. Initial Setup
```bash
# Make sure database has data
# Make sure .env file exists with credentials
# Install dependencies if needed:
pip install torch numpy python-dotenv

# Make script executable (Linux/Mac)
chmod +x train_models.py
```

### 2. First Training
```bash
# Start with lower requirements if you don't have much data yet
python train_models.py --min-data 100 --verbose
```

### 3. Regular Retraining
```bash
# Once you have more data, use standard settings
python train_models.py
```

## What Happens During Training

```
==================================================================
Training Model for Sensor 1
==================================================================
📊 Fetching data...
✓ Found 345 data points
✓ Split: 276 training, 69 validation
🔄 Preparing sequences (length=60)...
✓ Created 216 training sequences, 9 validation sequences
🏗️  Building model...
  - Input size: 3 (temp, diff, moving_avg)
  - Hidden size: 128
  - Layers: 3
  - Dropout: 0.2
  - Total parameters: 117,057
  - Trainable parameters: 117,057

🎯 Starting training (max 100 epochs)...
──────────────────────────────────────────────────────────────────
Epoch   1/100 | Train: 0.458123 | Val: 0.445234 | LR: 0.001000 | Patience: 0/15 ⭐
Epoch  10/100 | Train: 0.234567 | Val: 0.256789 | LR: 0.001000 | Patience: 0/15 ⭐
Epoch  20/100 | Train: 0.156789 | Val: 0.178901 | LR: 0.001000 | Patience: 0/15 ⭐
Epoch  30/100 | Train: 0.123456 | Val: 0.145678 | LR: 0.001000 | Patience: 2/15
Epoch  40/100 | Train: 0.098765 | Val: 0.134567 | LR: 0.000500 | Patience: 0/15 ⭐

🛑 Early stopping at epoch 52
✓ Best validation loss: 0.128945

✓ Restored best model (val_loss: 0.128945)

📈 Final Metrics:
  - Best validation loss: 0.128945
  - Final training loss: 0.092345
  - Epochs trained: 52

💾 Saving model...
  ✓ Model saved: models/model_1.pt
  ✓ Metadata saved: models/metadata_1.json
  ✓ Validation Loss: 0.128945

✅ Training completed successfully for sensor 1
```

## Understanding the Output

### Training Progress
- **Train Loss**: How well model fits training data (lower is better)
- **Val Loss**: How well model generalizes to new data (lower is better)
- **LR**: Learning rate (decreases when loss plateaus)
- **Patience**: Epochs without improvement (triggers early stopping at 15)
- **⭐**: Marks epochs that achieved best validation loss

### Early Stopping
The training stops automatically when the model stops improving. This prevents:
- Wasting time on unnecessary epochs
- Overfitting to training data
- Degrading model quality

### Best Model
The script automatically saves the model from the epoch with the best validation loss, not the last epoch.

## Interpreting Results

### Good Training
```
Val Loss: 0.128945
```
- Low validation loss (< 0.2) = Good predictions
- Val loss close to train loss = Good generalization

### Warning Signs

**Overfitting**:
```
Train: 0.050000 | Val: 0.450000
```
Big gap between train and val loss → Increase dropout or get more data

**Underfitting**:
```
Train: 0.850000 | Val: 0.870000
```
Both losses high → Increase model size or train longer

**Need More Data**:
```
❌ Not enough data: 45 points (need 200)
```
Wait for more data collection or use `--min-data 100`

## After Training

### Check Saved Models
```bash
ls -lh models/
```

Expected output:
```
metadata_1.json
model_1.pt
metadata_2.json
model_2.pt
...
```

### View Metadata
```bash
cat models/metadata_1.json
```

Example output:
```json
{
  "mean": 22.5,
  "std": 3.2,
  "val_loss": 0.128945,
  "saved_at": "2026-01-29T10:30:45.123456Z",
  "hidden_size": 128,
  "num_layers": 3,
  "sequence_size": 60,
  "dropout": 0.2
}
```

### Test the Model
The trained models will be automatically loaded when you start the Flask app:
```bash
python app.py
```

## Scheduling Automatic Training

### Using Cron (Linux/Mac)
```bash
# Edit crontab
crontab -e

# Add line to retrain daily at 2 AM
0 2 * * * cd /path/to/project && python train_models.py >> training.log 2>&1
```

### Using Windows Task Scheduler
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (e.g., daily at 2 AM)
4. Action: Start a program
5. Program: `python`
6. Arguments: `train_models.py`
7. Start in: `C:\path\to\project`

### Using systemd Timer (Linux)
```bash
# Create service file: /etc/systemd/system/temp-training.service
[Unit]
Description=Train Temperature Models

[Service]
Type=oneshot
WorkingDirectory=/path/to/project
ExecStart=/usr/bin/python3 train_models.py
User=youruser

# Create timer file: /etc/systemd/system/temp-training.timer
[Unit]
Description=Train models daily

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target

# Enable and start
sudo systemctl enable temp-training.timer
sudo systemctl start temp-training.timer
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'db'"
Make sure you're running the script from the correct directory:
```bash
cd /path/to/your/project
python train_models.py
```

### "Not enough data" for all sensors
You need to collect more temperature readings first. Options:
1. Wait for more data collection
2. Lower requirements temporarily: `python train_models.py --min-data 50`

### Training is very slow
1. Reduce epochs: `--epochs 50`
2. Check if you have GPU available
3. Reduce model size in the script (edit HIDDEN_SIZE, NUM_LAYERS)

### Models not loading in Flask app
Make sure model architecture matches. If you changed HIDDEN_SIZE, NUM_LAYERS, or SEQUENCE_SIZE, you need to retrain all models.

### Permission denied (Linux/Mac)
```bash
chmod +x train_models.py
```

## Best Practices

1. **Initial Training**: Use `--verbose` to watch progress
2. **Regular Updates**: Retrain weekly or daily for best accuracy
3. **Data Collection**: Collect at least 200 points before first training
4. **Monitoring**: Check validation loss - lower is better
5. **Backup**: Save your `models/` directory regularly

## Advanced: Batch Training Script

Create `batch_train.sh`:
```bash
#!/bin/bash
# Batch training with error handling

echo "Starting batch training..."
date

# Backup old models
if [ -d "models" ]; then
    cp -r models models_backup_$(date +%Y%m%d_%H%M%S)
fi

# Train models
python train_models.py --verbose 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log

# Check if successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
    # Optional: restart Flask server
    # systemctl restart temperature-server
else
    echo "Training failed!"
    exit 1
fi
```

Make it executable:
```bash
chmod +x batch_train.sh
./batch_train.sh
```
