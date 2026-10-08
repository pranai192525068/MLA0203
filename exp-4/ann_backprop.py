import numpy as np

# ---------------- DATASET ----------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# ---------------- ACTIVATION FUNCTION ----------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ---------------- INITIALIZE WEIGHTS ----------------
np.random.seed(1)

W1 = np.random.uniform(-1, 1, (2, 4))
W2 = np.random.uniform(-1, 1, (4, 1))

b1 = np.zeros((1, 4))
b2 = np.zeros((1, 1))

learning_rate = 0.5

# ---------------- BACKPROPAGATION ----------------
for epoch in range(10000):

    # Forward propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    output = sigmoid(final_input)

    # Calculate error
    error = Y - output

    # Backpropagation
    output_delta = error * sigmoid_derivative(output)

    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Update weights
    W2 += np.dot(hidden_output.T, output_delta) * learning_rate
    W1 += np.dot(X.T, hidden_delta) * learning_rate

    # Update bias
    b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

# ---------------- TESTING ----------------
print("Artificial Neural Network")
print("Using Backpropagation")
print("-----------------------------")

for i in range(len(X)):
    prediction = 1 if output[i][0] >= 0.5 else 0

    print("Input:", X[i],
          "Actual:", Y[i][0],
          "Predicted:", prediction)