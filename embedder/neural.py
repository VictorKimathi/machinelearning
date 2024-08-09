import numpy as np

# Define the input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Define the expected output data
y = np.array([[0], [1], [1], [0]])

# Define the neural network architecture
input_size = 2
hidden_size = 2
output_size = 1

# Initialize the weights randomly
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

# Define the activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the forward propagation function
def forward_propagation(X):
    layer1 = sigmoid(np.dot(X, W1) + b1)
    layer2 = sigmoid(np.dot(layer1, W2) + b2)
    return layer2

# Define the cost function (mean squared error)
def cost_function(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define the training function
def train(X, y, epochs=10000, learning_rate=0.1):
    global W1, b1, W2, b2
    for i in range(epochs):
        # Forward propagation
        layer1 = sigmoid(np.dot(X, W1) + b1)
        layer2 = sigmoid(np.dot(layer1, W2) + b2)

        # Compute the cost
        cost = cost_function(y, layer2)

        # Backpropagation
        delta2 = (layer2 - y) * layer2 * (1 - layer2)
        delta1 = np.dot(delta2, W2.T) * layer1 * (1 - layer1)

        # Update the weights
        W2 -= learning_rate * np.dot(layer1.T, delta2)
        b2 -= learning_rate * np.sum(delta2, axis=0)
        W1 -= learning_rate * np.dot(X.T, delta1)
        b1 -= learning_rate * np.sum(delta1, axis=0)

        if (i + 1) % 1000 == 0:
            print(f"Epoch [{i + 1}], Cost: {cost:.4f}")

# Train the neural network
train(X, y)

# Test the neural network
print("Predictions:")
print(forward_propagation(X))
