import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import mean_squared_error

X = load_digits().data
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

for q in [2, 5]:
    pca = PCA(n_components=q)
    Z = pca.fit_transform(X)
    reconstructed = pca.inverse_transform(Z)

    eig = pca.explained_variance_
    sigma2 = np.mean(eig[q:])
    W = pca.components_[:q].T * np.sqrt(
        np.maximum(eig[:q] - sigma2, 0)
    )

    C = W @ W.T + sigma2 * np.eye(X.shape[1])
    sign, logdet = np.linalg.slogdet(C)
    loglik = -0.5 * (
        X.shape[1] * np.log(2 * np.pi)
        + logdet
        + np.mean(np.sum(X @ np.linalg.inv(C) * X, axis=1))
    )

    fa = FactorAnalysis(n_components=q).fit(X)

    print("\nComponents:", q)
    print("W:\n", W)
    print("Sigma squared:", sigma2)
    print("PPCA Log Likelihood:", loglik)
    print("PCA Error:", mean_squared_error(X, reconstructed))
    print("FA Log Likelihood:", fa.score(X))

# Different noise in each feature
rng = np.random.default_rng(10)
Y = rng.normal(size=(500, 5))
Y[:, 1] *= 3
Y[:, 3] *= 2

p = PCA(n_components=2).fit(Y)
f = FactorAnalysis(n_components=2).fit(Y)

print("\nDifferent feature noise:")
print("PCA noise variance:", np.mean(p.explained_variance_[2:]))
print("FA noise variances:", f.noise_variance_)

plt.scatter(X[:, 0], X[:, 1], s=5)
plt.title("Digits Dataset - First Two Features")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()