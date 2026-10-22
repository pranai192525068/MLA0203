
from math import sqrt
from collections import Counter

# Training data
X_train = [
    [1, 2], [2, 3], [3, 3],
    [6, 5], [7, 7], [8, 6]
]
y_train = ["A", "A", "A", "B", "B", "B"]

# Test data
X_test = [[2, 2], [7, 6], [3, 4], [6, 6]]
y_test = ["A", "B", "A", "B"]

k = 3

def knn_predict(point):
    distances = []

    for i in range(len(X_train)):
        d = sqrt(
            (point[0] - X_train[i][0]) ** 2 +
            (point[1] - X_train[i][1]) ** 2
        )
        distances.append((d, y_train[i]))

    distances.sort(key=lambda x: x[0])
    nearest = [label for _, label in distances[:k]]
    return Counter(nearest).most_common(1)[0][0]

# Predict and display results
predictions = [knn_predict(point) for point in X_test]

print("Actual Output:", y_test)
print("Predicted Output:", predictions)

correct = sum(a == p for a, p in zip(y_test, predictions))
accuracy = correct / len(y_test)

print("Accuracy:", accuracy * 100, "%")
