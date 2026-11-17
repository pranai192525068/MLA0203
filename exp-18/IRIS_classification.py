
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Perceptron model
model = Perceptron(max_iter=1000, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

print("Predicted Values:", y_pred)
print("Actual Values:", y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100, "%")
