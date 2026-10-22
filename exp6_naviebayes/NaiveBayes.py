
import math
from collections import Counter

# Training data: [feature1, feature2]
X_train = [
    [1, 2], [2, 3], [2, 1],
    [6, 5], [7, 8], [8, 6]
]
y_train = ["A", "A", "A", "B", "B", "B"]

X_test = [[2, 2], [7, 6], [3, 2], [6, 7]]
y_test = ["A", "B", "A", "B"]

# Train Gaussian Naive Bayes
classes = sorted(set(y_train))
model = {}

for c in classes:
    rows = [X_train[i] for i in range(len(y_train))
            if y_train[i] == c]
    means = [sum(col) / len(col)
             for col in zip(*rows)]
    variances = [
        sum((row[j] - means[j]) ** 2 for row in rows)
        / len(rows) + 1e-9
        for j in range(len(means))
    ]
    model[c] = (len(rows) / len(y_train), means, variances)

def predict(x):
    scores = {}
    for c, (prior, means, variances) in model.items():
        score = math.log(prior)
        for j in range(len(x)):
            score += -0.5 * math.log(2 * math.pi * variances[j])
            score -= (x[j] - means[j]) ** 2 / (2 * variances[j])
        scores[c] = score
    return max(scores, key=scores.get)

# Test and evaluate
predictions = [predict(x) for x in X_test]

print("Actual Output:", y_test)
print("Predicted Output:", predictions)

labels = classes
matrix = [[0 for _ in labels] for _ in labels]

for actual, predicted in zip(y_test, predictions):
    matrix[labels.index(actual)][labels.index(predicted)] += 1

print("\nConfusion Matrix:")
print("     ", labels)
for label, row in zip(labels, matrix):
    print(label, "   ", row)

accuracy = sum(a == p for a, p in zip(y_test, predictions)) / len(y_test)
print("\nAccuracy:", round(accuracy * 100, 2), "%")
