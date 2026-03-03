import os
import torch.nn as nn
import numpy as np

def ensure_models_directory(models_dir="models"):
    """Ensure the models directory exists"""
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"✓ Created models directory: {models_dir}")

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
