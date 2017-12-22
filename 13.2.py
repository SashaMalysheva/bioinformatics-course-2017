import numpy as np
N = 2  # open
M = 2  # hidden
T = 12  # sequence length

def FB(pi, A, B, sequence):
    result = np.zeros((T, N))
    alpha = np.zeros((T, N))
    beta = np.zeros((T, N))
    for i in range(N):
        alpha[0, i] = pi[i] * B[i, sequence[0]]
    for i in range(T):
        for j in range(N):
            for g in range(N):
                alpha[i, j] += alpha[i - 1, g] *A[g, j] * B[j, sequence[i]]

    for i in range(N):
        beta[T-1, i] = 1
    for i in range(T - 2, -1, -1):
        for j in range(N):
            for g in range(N):
                beta[i, j] += beta[i + 1, g] * A[j, g] * B[g, sequence[i + 1]]

    sum = 0
    for i in range(N):
        sum += alpha[T-1, i]
    for i in range(T):
        for j in range(N):
            result[i,j] = alpha[i,j] * beta[i,j] / sum
    return result

pi = np.array([0.5, 0.5])
A = np.array([[0.5, 0.5], [0.5, 0.5]])
B = np.array([[0.9, 0.1], [0.5, 0.5]])
sequence = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])

res = FB(pi, A, B, sequence)
print(res)