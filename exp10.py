# prompt: implement LSTM model for time series forcasting also visulize it

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate some sample time series data
def create_time_series_data(length, trend=0.1, seasonality=0.5, noise=0.1):
  time_series = []
  for i in range(length):
    value = trend * i + seasonality * np.sin(i) + np.random.randn() * noise
    time_series.append(value)
  return np.array(time_series)

# Create training data
time_series_data = create_time_series_data(100)

# Prepare data for LSTM (look back one time step)
def create_dataset(dataset, look_back=1):
  X, Y = [], []
  for i in range(len(dataset) - look_back - 1):
    a = dataset[i:(i + look_back), 0]
    X.append(a)
    Y.append(dataset[i + look_back, 0])
  return np.array(X), np.array(Y)

look_back = 1
X, Y = create_dataset(time_series_data.reshape(-1, 1), look_back)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Create and train the LSTM model
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=100, batch_size=1, verbose=2)

# Make predictions
trainPredict = model.predict(X)

# Plot the results
plt.plot(time_series_data, label='Original')
plt.plot(np.arange(look_back, len(trainPredict) + look_back), trainPredict, label='Predictions')
plt.legend()
plt.show()