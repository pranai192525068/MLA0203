
import numpy as np
from sklearn.mixture import GaussianMixture

# Sample dataset
X = np.array([
    [1.0], [1.5], [2.0],
    [8.0], [8.5], [9.0]
])

# Implement EM using Gaussian Mixture Model
model = GaussianMixture(
    n_components=2,
    random_state=42,
    n_init=10
)

model.fit(X)

# Predict clusters
labels = model.predict(X)

print("Data Points:", X.flatten())
print("Cluster Labels:", labels)
print("Means:", np.round(model.means_.flatten(), 2))
print("Converged:", model.converged_)
