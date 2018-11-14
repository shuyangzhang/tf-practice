# coding  = utf-8

import numpy as np

nan = np.nan
T = np.array([
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
    [[0.0, 1.0, 0.0], [nan, nan, nan], [0.0, 0.0, 1.0]],
    [[nan, nan, nan], [0.8, 0.1, 0.1], [nan, nan, nan]]
    ]
)

R = np.array([
    [[10., 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[10., 0.0, 0.0], [nan, nan, nan], [0.0, 0.0, -50.]],
    [[nan, nan, nan], [40., 0.0, 0.0], [nan, nan, nan]]
    ]
)

possible_action = [[0, 1, 2], [0, 2], [1]]

Q = np.full((3,3), -np.inf)
for state, action in enumerate(possible_action):
    Q[state, action] = 0.0

learning_rate = 0.01
discount_rate = 0.95
n_iterations = 100

for iteration in range(n_iterations):
    Q_prev = Q.copy()
    for s in range(3):
        for a in possible_action[s]:
            Q[s, a] = np.sum(
                [
                    T[s, a, sp] * (R[s, a, sp] + discount_rate * np.max(Q_prev[sp]))
                    for sp in range(3)
                ]
            )


print(Q)
print(np.argmax(Q, axis=1))
