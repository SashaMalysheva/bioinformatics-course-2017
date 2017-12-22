'''
A = np.array([[1.0, 0.0], [0.0, 1.0]])
'''

import numpy as np

N = 2  # open
M = 2  # hidden
T = 12  # sequence length

pi = np.array([0.5, 0.5])
A = np.array([[1.0, 0.0], [0.0, 1.0]])
B = np.array([[0.9, 0.1], [0.5, 0.5]])
sequence = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])

delta = np.zeros((T, M))
index = np.zeros((T, M))
result = np.zeros((T))

index[0, :] = np.arange(M)

for i in range(M):
    delta[0, i] = pi[i] * B[i, sequence[0]]

for t in range(T - 1):
    for j in range(M):
        for i in range(M):
            if delta[t, i] * A[i, j] * B[j, sequence[t + 1]] > delta[t + 1, j]:
                delta[t + 1, j] = delta[t, i] * A[i, j] * B[j, sequence[t + 1]]
                index[t + 1, j] = i


last_index = np.argmax(delta[T - 1, :])
result[T - 1] = last_index

for t in range(T - 1, 0, -1):
    last_index = index[t, int(last_index)]
    result[t - 1] = last_index

print(result)
