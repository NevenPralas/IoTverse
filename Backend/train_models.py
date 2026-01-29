#!/usr/bin/env python3
"""
Standalone Training Script for Temperature Prediction Models

This script trains the LSTM models for all sensors without running the Flask server.
Useful for:
- Initial model training
- Retraining after data collection
- Batch training updates
- Testing model improvements

Usage:
    python train_models.py                    # Train all sensors
    python train_models.py --sensor 1         # Train specific sensor
    python train_models.py --epochs 150       # Custom epochs
    python train_models.py --verbose          # Detailed output
"""

import os
import argparse
import datetime
import torch
import torch.nn as nn
import numpy as np
import json
from dotenv import load_dotenv
import db


# ===== CONFIGURATION ==========================================================
MODELS_DIR = "models"
SENSOR_IDS = [1, 2, 3, 4]
REQ_DATA_POINTS = 200
SEQUENCE_SIZE = 60
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.2


# ===== LSTM MODEL =============================================================
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


# ===== HELPER FUNCTIONS =======================================================
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

    # Calculate rate of change
    temp_diff = np.diff(temperatures, prepend=temperatures[0])
    
    # Calculate moving averages
    window_size = min(7, len(temperatures) // 4)
    if window_size > 1:
        moving_avg = np.convolve(temperatures, np.ones(window_size)/window_size, mode='same')
    else:
        moving_avg = temperatures.copy()

    # Normalize data
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

    return np.array(X), np.array(y), temp_mean, temp_std


def save_model(model, sensor_id, mean, std, val_loss, hyperparams):
    """Save the trained model and metadata"""
    try:
        model_path = os.path.join(MODELS_DIR, f"model_{sensor_id}.pt")
        torch.save(model.state_dict(), model_path)

        metadata_path = os.path.join(MODELS_DIR, f"metadata_{sensor_id}.json")
        metadata = {
            'mean': float(mean),
            'std': float(std),
            'val_loss': float(val_loss) if val_loss is not None else None,
            'saved_at': datetime.datetime.now(datetime.UTC).isoformat(),
            'hidden_size': hyperparams['hidden_size'],
            'num_layers': hyperparams['num_layers'],
            'sequence_size': hyperparams['sequence_size'],
            'dropout': hyperparams['dropout']
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Model saved: {model_path}")
        print(f"  ✓ Metadata saved: {metadata_path}")
        if val_loss is not None:
            print(f"  ✓ Validation Loss: {val_loss:.6f}")
        return True
    except Exception as e:
        print(f"  ❌ Error saving model: {e}")
        return False


# ===== TRAINING FUNCTION ======================================================
def train_model(sensor_id, max_epochs=100, verbose=False):
    """
    Train a model for a specific sensor
    
    Args:
        sensor_id: Sensor identifier
        max_epochs: Maximum training epochs
        verbose: Print detailed progress
    """
    print(f"\n{'='*70}")
    print(f"Training Model for Sensor {sensor_id}")
    print(f"{'='*70}")

    # Fetch data
    print(f"📊 Fetching data...")
    data = db.get_recent_temperature_data(sensor_id, limit=500)
    
    if len(data) < REQ_DATA_POINTS:
        print(f"❌ Not enough data: {len(data)} points (need {REQ_DATA_POINTS})")
        return False

    print(f"✓ Found {len(data)} data points")

    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"✓ Split: {len(train_data)} training, {len(val_data)} validation")

    # Prepare sequences
    print(f"🔄 Preparing sequences (length={SEQUENCE_SIZE})...")
    train_result = prepare_sequences(train_data, SEQUENCE_SIZE)
    if train_result is None:
        print(f"❌ Cannot prepare training sequences")
        return False

    X_train, y_train, mean, std = train_result

    val_result = prepare_sequences(val_data, SEQUENCE_SIZE)
    if val_result is None:
        print(f"⚠️  Not enough validation data, using training data only")
        X_val, y_val = None, None
    else:
        X_val, y_val, _, _ = val_result
        print(f"✓ Created {len(X_train)} training sequences, {len(X_val)} validation sequences")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
    
    if X_val is not None:
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1)

    # Create model
    print(f"🏗️  Building model...")
    print(f"  - Input size: 3 (temp, diff, moving_avg)")
    print(f"  - Hidden size: {HIDDEN_SIZE}")
    print(f"  - Layers: {NUM_LAYERS}")
    print(f"  - Dropout: {DROPOUT}")
    
    model = TemperatureLSTM(
        input_size=3,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15
    best_model_state = None

    batch_size = 16
    
    print(f"\n🎯 Starting training (max {max_epochs} epochs)...")
    print(f"{'─'*70}")

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        num_batches = 0

        # Shuffle training data
        indices = torch.randperm(len(X_train_tensor))
        X_train_shuffled = X_train_tensor[indices]
        y_train_shuffled = y_train_tensor[indices]

        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_shuffled[i:i + batch_size]
            batch_y = y_train_shuffled[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / num_batches

        # Validation phase
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
        else:
            val_loss = avg_train_loss

        # Update learning rate
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            improvement_marker = " ⭐"
        else:
            patience_counter += 1
            improvement_marker = ""

        # Print progress
        if verbose or (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1:3d}/{max_epochs} | "
                  f"Train: {avg_train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Patience: {patience_counter}/{max_patience}{improvement_marker}")

        # Early stopping
        if patience_counter >= max_patience:
            print(f"\n🛑 Early stopping at epoch {epoch + 1}")
            print(f"✓ Best validation loss: {best_val_loss:.6f}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Restored best model (val_loss: {best_val_loss:.6f})")

    model.eval()

    # Calculate final metrics
    print(f"\n📈 Final Metrics:")
    print(f"  - Best validation loss: {best_val_loss:.6f}")
    print(f"  - Final training loss: {avg_train_loss:.6f}")
    print(f"  - Epochs trained: {epoch + 1}")
    
    # Save model
    print(f"\n💾 Saving model...")
    hyperparams = {
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'sequence_size': SEQUENCE_SIZE,
        'dropout': DROPOUT
    }
    success = save_model(model, sensor_id, mean, std, best_val_loss, hyperparams)
    
    if success:
        print(f"\n✅ Training completed successfully for sensor {sensor_id}")
    
    return success


# ===== MAIN SCRIPT ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM models for temperature prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_models.py                     # Train all sensors
  python train_models.py --sensor 1          # Train sensor 1 only
  python train_models.py --sensor 1 2 3      # Train sensors 1, 2, and 3
  python train_models.py --epochs 150        # Train for max 150 epochs
  python train_models.py --verbose           # Show detailed progress
  python train_models.py -s 1 -e 200 -v      # All options combined
        """
    )
    
    parser.add_argument(
        '-s', '--sensor',
        type=int,
        nargs='+',
        help='Sensor ID(s) to train. If not specified, trains all sensors.'
    )
    
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=100,
        help='Maximum number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed training progress for each epoch'
    )
    
    parser.add_argument(
        '--min-data',
        type=int,
        default=REQ_DATA_POINTS,
        help=f'Minimum data points required (default: {REQ_DATA_POINTS})'
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Determine which sensors to train
    if args.sensor:
        sensors_to_train = args.sensor
    else:
        sensors_to_train = SENSOR_IDS

    # Update global config if specified
    # global REQ_DATA_POINTS
    # REQ_DATA_POINTS = args.min_data

    print("=" * 70)
    print("Temperature Prediction Model - Training Script")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Sensors to train: {sensors_to_train}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Sequence size: {SEQUENCE_SIZE}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Min data points: {REQ_DATA_POINTS}")
    print(f"  Verbose: {args.verbose}")

    # Ensure directories exist
    ensure_models_directory()

    # Initialize database
    print(f"\n🗄️  Initializing database...")
    db.init_database()
    print(f"✓ Database ready")

    # Train each sensor
    results = {}
    start_time = datetime.datetime.now()

    for sensor_id in sensors_to_train:
        sensor_start = datetime.datetime.now()
        success = train_model(sensor_id, max_epochs=args.epochs, verbose=args.verbose)
        sensor_end = datetime.datetime.now()
        duration = (sensor_end - sensor_start).total_seconds()
        
        results[sensor_id] = {
            'success': success,
            'duration': duration
        }

    # Print summary
    end_time = datetime.datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    
    successful = sum(1 for r in results.values() if r['success'])
    failed = len(results) - successful
    
    print(f"\nResults:")
    for sensor_id, result in results.items():
        status = "✅ Success" if result['success'] else "❌ Failed"
        duration_str = f"{result['duration']:.1f}s"
        print(f"  Sensor {sensor_id}: {status} ({duration_str})")
    
    print(f"\nOverall:")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print(f"  Total time: {total_duration:.1f}s")
    
    if successful == len(results):
        print(f"\n🎉 All models trained successfully!")
    elif successful > 0:
        print(f"\n⚠️  Some models failed to train")
    else:
        print(f"\n❌ All models failed to train")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
