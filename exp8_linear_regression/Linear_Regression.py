
# Linear Regression Algorithm

X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]

n = len(X)

# Calculate mean
mean_x = sum(X) / n
mean_y = sum(Y) / n

# Calculate slope
numerator = sum(
    (X[i] - mean_x) * (Y[i] - mean_y)
    for i in range(n)
)

denominator = sum(
    (X[i] - mean_x) ** 2
    for i in range(n)
)

slope = numerator / denominator

# Calculate intercept
intercept = mean_y - slope * mean_x

# Predict output
x_test = 6
y_pred = intercept + slope * x_test

print("Slope:", round(slope, 2))
print("Intercept:", round(intercept, 2))
print("Input value:", x_test)
print("Predicted output:", round(y_pred, 2))
