import numpy as np
import matplotlib.pyplot as plt

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def train(sarsa=False, epsilon=0.1, alpha=0.5, decay=False):
    Q = np.zeros((48, 4))
    returns = []

    for ep in range(500):
        state = (3, 0)
        total = 0
        eps = max(0.01, 1 - ep / 500) if decay else epsilon

        for step in range(500):
            s = state[0] * 12 + state[1]

            if np.random.rand() < eps:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[s])

            r, c = state
            dr, dc = ACTIONS[action]
            nr = max(0, min(3, r + dr))
            nc = max(0, min(11, c + dc))
            next_state = (nr, nc)

            reward = -1

            if nr == 3 and 1 <= nc <= 10:
                reward = -100
                next_state = (3, 0)

            done = next_state == (3, 11)
            ns = next_state[0] * 12 + next_state[1]

            if sarsa:
                next_action = (np.random.randint(4) if np.random.rand() < eps
                               else np.argmax(Q[ns]))
                target = reward + (0 if done else 0.9 * Q[ns, next_action])
            else:
                target = reward + (0 if done else 0.9 * np.max(Q[ns]))

            Q[s, action] += alpha * (target - Q[s, action])
            state = next_state
            total += reward

            if done:
                break

        returns.append(total)

    return Q, returns

def show_path(Q, title):
    state = (3, 0)
    path = [state]

    for _ in range(50):
        s = state[0] * 12 + state[1]
        action = np.argmax(Q[s])
        dr, dc = ACTIONS[action]

        nr = max(0, min(3, state[0] + dr))
        nc = max(0, min(11, state[1] + dc))

        if nr == 3 and 1 <= nc <= 10:
            state = (3, 0)
        else:
            state = (nr, nc)

        path.append(state)

        if state == (3, 11):
            break

    print(title, "Path:", path)

# Test cases
for alpha in [0.5, 0.1]:
    for decay in [False, True]:
        for sarsa in [False, True]:
            Q, rewards = train(sarsa, 0.1, alpha, decay)

            print("\nAlgorithm:", "SARSA" if sarsa else "Q-learning")
            print("Alpha:", alpha, "Decay:", decay)
            print("Average return:", np.mean(rewards))

            show_path(Q, "Learned")

            plt.plot(rewards, label=
                     "SARSA" if sarsa else "Q-learning")

        plt.title("Cliff Walking Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend()
        plt.show()