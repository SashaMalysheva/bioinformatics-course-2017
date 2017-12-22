import numpy as np


def alig(s, t):
    s1 = "a" + s + "a"
    s2 = "a" + t + "a"
    t1 = len(s1)
    t2 = len(s2)
    alpha = np.zeros((t1, t2, 3))
    beta = np.zeros((t1, t2, 3))

    pi = np.array([1 / 3, 1 / 3, 1 / 3])
    A = np.array([[0.8, 0.1, 0.1, 0], [0.6, 0.3, 0.1, 0], [0.6, 0.1, 0.3, 0]])
    B = np.array([[0.01, 0.99], [0.5, 0.5], [0.5, 0.5]])
    alpha[0, 0, 0] = 1

    for i in range(1, t1):
        for h in range(3):
            alpha[i, 0, 2] += alpha[i - 1, 0, h] * A[h, 2] * B[2, int(s1[i] == s2[0])]

    for j in range(1, t2):
        for h in range(3):
            alpha[0, j, 1] += alpha[0, j - 1, h] * A[h, 1] * B[1, int(s1[0] == s2[j])]

    for i in range(1, t1):
        for j in range(1, t2):
            for h in range(3):
                alpha[i, j, 0] += alpha[i - 1, j - 1, h] * A[h, 0] * B[0, int(s1[i] == s2[j])]
                alpha[i, j, 1] += alpha[i, j - 1, h] * A[h, 1] * B[1, int(s1[i] == s2[j])]
                alpha[i, j, 2] += alpha[i - 1, j, h] * A[h, 2] * B[2, int(s1[i] == s2[j])]
            if i != t1 - 1:
                alpha[i, t2 - 1, 0] = 0
            alpha[i, t2 - 1, 1] = 0
            alpha[i, t2 - 1, 2] = 0

            if j != t2 - 1:
                alpha[t1 - 1, j, 0] = 0
            alpha[t1 - 1, j, 1] = 0
            alpha[t1 - 1, j, 2] = 0

    beta[t1 - 1, t2 - 1, :] = 1

    for i in range(t1 - 2, -1, -1):
        for h in range(3):
            beta[i, t2 - 1, 2] += beta[i + 1, t2 - 1, h] * A[2, h] * B[h, int(s1[i + 1] == s2[t2 - 1])]

    for j in range(t2 - 2, -1, -1):
        for h in range(3):
            beta[t1 - 1, j, 1] += beta[t1 - 1, j + 1, h] * A[1, h] * B[h, int(s2[j + 1] == s1[t1 - 1])]

    for i in range(t1 - 2, -1, -1):
        for j in range(t2 - 2, -1, -1):
            for h in range(3):
                beta[i, j, h] += beta[i + 1, j + 1, 0] * A[h, 0] * B[0, int(s1[i + 1] == s2[j + 1])]
                beta[i, j, h] += beta[i, j + 1, 1] * A[h, 1] * B[1, int(s1[i] == s2[j + 1])]
                beta[i, j, h] += beta[i + 1, j, 2] * A[h, 2] * B[2, int(s1[i + 1] == s2[j])]
            if i != 0:
                beta[i, 0, 0] = 0
            if j != 0:
                beta[0, j, 0] = 0


    result = np.multiply(alpha[:, :, 0], beta[:, :, 0]) / alpha[t1 - 1, t2 - 1, 0]
    a = np.multiply(alpha[:, :, 1], beta[:, :, 1]) / alpha[t1 - 1, t2 - 1, 0]
    b = np.multiply(alpha[:, :, 2], beta[:, :, 2]) / alpha[t1 - 1, t2 - 1, 0]
    print(result[1:, 1:])

    return result


s = "AGAGA"
t = "AGA"

alig(s, t)
