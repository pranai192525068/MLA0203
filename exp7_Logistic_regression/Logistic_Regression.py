
import math

# Training data
X_train = [1, 2, 3, 4, 5, 6]
y_train = [0, 0, 0, 1, 1, 1]

# Logistic Regression parameters
w = 0.0
b = 0.0
learning_rate = 0.1

# Train the model using gradient descent
for epoch in range(2000):
    dw = 0.0
    db = 0.0
    n = len(X_train)

    for x, y in zip(X_train, y_train):
        z = w * x + b
        z = max(-500, min(500, z))
        prediction = 1 / (1 + math.exp(-z))

        dw += (prediction - y) * x
        db += prediction - y

    w -= learning_rate * dw / n
    b -= learning_rate * db / n

# Predict output
X_test = [1.5, 2.5, 4.5, 5.5]
y_test = [0, 0, 1, 1]
predictions = []

for x in X_test:
    probability = 1 / (1 + math.exp(-(w * x + b)))
    predictions.append(1 if probability >= 0.5 else 0)

print("Actual Output:", y_test)
print("Predicted Output:", predictions)

correct = sum(a == p for a, p in zip(y_test, predictions))
accuracy = correct / len(y_test) * 100

print("Accuracy:", accuracy, "%")
