from __future__ import print_function, division
import numpy as np

gap = -3
match = 1.
labels = {'A': 0, 'C': 1, 'T': 2, 'G': 3}
mismatch = [[+1, -100, -3, -4], [0, +2, -3, -4], [0, 0, +1, -4], [0, 0, 0, +2]]
s, t = 'ATCCGAGGTC', 'AGTCGCTGTC'


def _match(alpha, beta):
    if alpha == '-' or beta == '-':
        return gap
    else:
        return mismatch[min(labels[alpha],labels[beta])][max(labels[alpha],labels[beta])]


def needl(s, t):
    dim_i, dim_j = len(s), len(t)
    m = np.zeros((dim_i + 1, dim_j + 1))
    pointer = np.zeros((dim_i + 1, dim_j + 1))  # to store the traceback path

    # Initialization
    for i in range(dim_i + 1):
        m[i, 0] = gap * i
    for j in range(dim_j + 1):
        m[0, j] = gap * j

    for i in range(1, dim_i + 1):
        for j in range(1, dim_j + 1):
            diag = m[i - 1, j - 1] + _match(s[i - 1], t[j - 1])
            left = m[i - 1, j] + gap
            up = m[i, j - 1] + gap
            m[i, j] = max(diag, left, up)
            if m[i, j] == 0:
                pointer[i, j] = 0
            if m[i, j] == left:
                pointer[i, j] = 1
            if m[i, j] == up:
                pointer[i, j] = 2
            if m[i, j] == diag:
                pointer[i, j] = 3

    align1, align2 = '', ''
    i, j = dim_i, dim_j

    # Traceback
    i = dim_i
    j = dim_j
    while pointer[i, j] != 0:
        if pointer[i, j] == 3:
            align1 += s[i - 1]
            align2 += t[j - 1]
            i -= 1
            j -= 1
        elif pointer[i, j] == 2:
            align1 += '-'
            align2 += t[j - 1]
            j -= 1
        elif pointer[i, j] == 1:
            align1 += s[i - 1]
            align2 += '-'
            i -= 1

    print(align1[::-1])
    print(align2[::-1])


needl(s, t)