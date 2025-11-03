import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from database import get_all_data

data = get_all_data()['temperature'].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)


X, y = create_sequences(data_scaled, seq_len=10)


# Model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=20, batch_size=16, verbose=1)


# Prediction
last_sequence = data_scaled[-10:]
pred = model.predict(np.expand_dims(last_sequence, axis=0))
pred_temp = scaler.inverse_transform(pred)[0][0]

print(f"Next predicted temperature value: {
      pred_temp:.2f} °C")
