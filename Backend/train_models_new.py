#!/usr/bin/env python3
"""
FIXED: Standalone Training Script for Temperature Prediction Models
Key fix: Saves ALL normalization parameters needed for inference
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

# Import extracted components
from model import TemperatureLSTM, prepare_sequences, ensure_models_directory

# ===== CONFIGURATION ==========================================================
MODELS_DIR = "models"
SENSOR_IDS = [1, 2, 3, 4]
REQ_DATA_POINTS = 200
SEQUENCE_SIZE = 60
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.2


# ===== HELPER FUNCTIONS =======================================================
def save_model(model, sensor_id, norm_params, val_loss, hyperparams):
    """FIXED: Save model with ALL normalization parameters"""
    try:
        model_path = os.path.join(MODELS_DIR, f"model_{sensor_id}.pt")
        torch.save(model.state_dict(), model_path)

        metadata_path = os.path.join(MODELS_DIR, f"metadata_{sensor_id}.json")
        metadata = {
            # FIXED: Save all normalization parameters
            'norm_params': norm_params,
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
        print(f"  ✓ Normalization params: {norm_params}")
        if val_loss is not None:
            print(f"  ✓ Validation Loss: {val_loss:.6f}")
        return True
    except Exception as e:
        print(f"  ✗ Error saving model: {e}")
        return False


# ===== TRAINING FUNCTION ======================================================
def train_model(sensor_id, max_epochs=100, verbose=False):
    """Train a model for a specific sensor"""
    print(f"\n{'=' * 70}")
    print(f"Training Model for Sensor {sensor_id}")
    print(f"{'=' * 70}")

    # Fetch data
    print(f"Fetching data...")
    data = db.get_recent_temperature_data(sensor_id, limit=500)

    if len(data) < REQ_DATA_POINTS:
        print(f"Not enough data: {len(data)} points (need {REQ_DATA_POINTS})")
        return False

    print(f"Found {len(data)} data points")

    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Split: {len(train_data)} training, {len(val_data)} validation")

    # Prepare sequences
    print(f"Preparing sequences (length={SEQUENCE_SIZE})...")
    train_result = prepare_sequences(train_data, SEQUENCE_SIZE)
    if train_result is None:
        print(f"Cannot prepare training sequences")
        return False

    X_train, y_train, norm_params = train_result

    val_result = prepare_sequences(val_data, SEQUENCE_SIZE)
    if val_result is None:
        print(f"Not enough validation data, using training data only")
        X_val, y_val = None, None
    else:
        X_val, y_val, _ = val_result  # Don't need norm_params from validation
        print(f"Created {len(X_train)} training sequences, {len(X_val)} validation sequences")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)

    if X_val is not None:
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1)

    # Create model
    print(f"Building model...")
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

    print(f"\nStarting training (max {max_epochs} epochs)...")
    print(f"{'─' * 70}")

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
            print(f"\nEarly stopping at epoch {epoch + 1}")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Restored best model (val_loss: {best_val_loss:.6f})")

    model.eval()

    # Calculate final metrics
    print(f"\nFinal Metrics:")
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
    success = save_model(model, sensor_id, norm_params, best_val_loss, hyperparams)

    if success:
        print(f"\n✓ Training completed successfully for sensor {sensor_id}")

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

    print("=" * 70)
    print("Temperature Prediction Model - Training Script (FIXED)")
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
    ensure_models_directory(MODELS_DIR)

    # Initialize database
    print(f"\nInitializing database...")
    db.init_database()
    print(f"Database ready")

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
        status = "✓ Success" if result['success'] else "✗ Failed"
        duration_str = f"{result['duration']:.1f}s"
        print(f"  Sensor {sensor_id}: {status} ({duration_str})")

    print(f"\nOverall:")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print(f"  Total time: {total_duration:.1f}s")

    if successful == len(results):
        print(f"\n✓ All models trained successfully!")
    elif successful > 0:
        print(f"\n⚠ Some models failed to train")
    else:
        print(f"\n✗ All models failed to train")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()