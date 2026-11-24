import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def halving(n):
    # -1 means don't care, 0 and 1 are conditions
    H = list(product([-1, 0, 1], repeat=n))

    target = H[np.random.randint(len(H))]
    mistakes = 0

    for _ in range(50):
        x = np.random.randint(0, 2, n)

        y = int(all(a == -1 or a == b
                    for a, b in zip(target, x)))

        predictions = [
            int(all(a == -1 or a == b
                    for a, b in zip(h, x)))
            for h in H
        ]

        pred = int(sum(predictions) > len(H) / 2)

        if pred != y:
            mistakes += 1

        H = [h for h, p in zip(H, predictions) if p == y]

    return mistakes, len(H)

# Halving tests
for n in [3, 5]:
    mistakes, remaining = halving(n)
    print("\nn =", n)
    print("Mistakes:", mistakes)
    print("Theoretical bound:", n * np.log2(3))

# Weighted Majority
np.random.seed(10)
N = 10
beta = 0.5
weights = np.ones(N)

labels = np.random.randint(0, 2, 50)
experts = np.array([
    [labels[i] if np.random.rand() > 0.2
     else 1-labels[i] for i in range(50)]
    for _ in range(N)
])

mistakes = 0
history = []

for i in range(50):
    prediction = int(
        np.sum(weights * experts[:, i]) >=
        np.sum(weights) / 2
    )

    if prediction != labels[i]:
        mistakes += 1

    weights[experts[:, i] != labels[i]] *= beta
    history.append(mistakes)

best = min(np.sum(experts != labels, axis=1))
bound = (np.log(N) + best * np.log(1/beta)) / np.log(2/(1+beta))

print("\nWeighted Majority")
print("Mistakes:", mistakes)
print("Best expert mistakes:", best)
print("Theoretical bound:", round(bound, 2))

plt.plot(history)
plt.title("Weighted Majority Mistakes")
plt.xlabel("Number of Trials")
plt.ylabel("Cumulative Mistakes")
plt.show()