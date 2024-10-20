import numpy as np

# Helper function: Activation functions and their derivatives
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return np.where(Z > 0, 1, 0)

# Helper function: Initialize parameters for a deep neural network
def initialize_parameters(layers_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

# Helper function: Forward propagation
def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the network

    for l in range(1, L):
        A_prev = A
        Z = np.dot(parameters['W' + str(l)], A_prev) + parameters['b' + str(l)]
        A = relu(Z)
        caches.append((A_prev, Z, parameters['W' + str(l)], parameters['b' + str(l)]))

    # Output layer
    ZL = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)]
    AL = sigmoid(ZL)
    caches.append((A, ZL, parameters['W' + str(L)], parameters['b' + str(L)]))

    return AL, caches

# Helper function: Compute the cost
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(1/m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    return np.squeeze(cost)

# Helper function: Backward propagation
def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)  # number of layers
    m = AL.shape[1]

    # Initial gradient on the output layer
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Output layer gradients
    current_cache = caches[L - 1]
    A_prev, ZL, W, b = current_cache
    dZL = dAL * sigmoid_derivative(ZL)
    grads["dW" + str(L)] = (1/m) * np.dot(dZL, A_prev.T)
    grads["db" + str(L)] = (1/m) * np.sum(dZL, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZL)

    # Backpropagation for hidden layers
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        A_prev, Z, W, b = current_cache
        dZ = dA_prev * relu_derivative(Z)
        grads["dW" + str(l + 1)] = (1/m) * np.dot(dZ, A_prev.T)
        grads["db" + str(l + 1)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

    return grads

# Helper function: Update parameters using gradient descent
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters

# Training the neural network
def model(X, Y, layers_dims, learning_rate=0.01, num_iterations=10000):
    np.random.seed(1)
    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):
        # Forward propagation
        AL, caches = forward_propagation(X, parameters)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = backward_propagation(AL, Y, caches)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost}")

    return parameters

# Example: Train a 3-layer neural network (2 hidden layers)
layers_dims = [3, 4, 4, 1]  # Input layer: 3 units, two hidden layers with 4 units each, output layer with 1 unit

# Example input data (X) and labels (Y)
X = np.random.randn(3, 5)  # 3 features, 5 examples
Y = np.array([[1, 0, 1, 0, 1]])  # Corresponding labels

# Train the model
parameters = model(X, Y, layers_dims, learning_rate=0.01, num_iterations=10000)
