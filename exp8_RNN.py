import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing import sequence

# For reproducibility
np.random.seed(42)
# Load the IMDb dataset
max_features = 10000  # Use only the top 10,000 words
max_len = 500         # Cut off reviews after 500 words

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure they have the same length
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

print(f'x_train shape: {x_train.shape}, y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape}, y_test shape: {y_test.shape}')
# Build the RNN model
model = Sequential()

# Embedding layer: Maps each word index to a vector of size 32
model.add(Embedding(max_features, 32, input_length=max_len))

# SimpleRNN layer
model.add(SimpleRNN(32, return_sequences=False))  # return_sequences=False for classification

# Dense output layer with a sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())
# Train the model
batch_size = 64
epochs = 5

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test))
# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test Accuracy: {test_acc:.4f}')
