import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
  return x * (1 - x)

# Input dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Output dataset
y = np.array([[0], [1], [1], [0]])

# Define the number of neurons in each layer
input_layer_neurons = 2
hidden_layer_neurons = 2
output_layer_neurons = 1

# Initialize weights and biases with random values
hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
output_bias = np.random.uniform(size=(1, output_layer_neurons))

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training the model
for epoch in range(epochs):
  # Forward propagation
  hidden_layer_activation = np.dot(X, hidden_weights) + hidden_bias
  hidden_layer_output = sigmoid(hidden_layer_activation)
  output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
  predicted_output = sigmoid(output_layer_activation)

  # Backpropagation
  error = y - predicted_output
  d_predicted_output = error * sigmoid_derivative(predicted_output)
  error_hidden_layer = d_predicted_output.dot(output_weights.T)
  d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

  # Update weights and biases
  output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
  output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
  hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
  hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Test the model
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n", predicted_output)
