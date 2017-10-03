from __future__ import print_function, division
import numpy as np

gap = -1
match = 1.
labels = dict
mismatch = -1.5


def _match(alpha, beta):
    if alpha == beta:
        return match
    elif alpha == '-' or beta == '-':
        return gap
    else:
        return mismatch


def water(s, t):
    dim_i, dim_j = len(s), len(t)
    m = np.zeros((dim_i + 1, dim_j + 1))
    pointer = np.zeros((dim_i + 1, dim_j + 1))
    # Initialization
    max_m = 0
    for i in range(1, dim_i + 1):
        for j in range(1, dim_j + 1):
            diag = m[i - 1, j - 1] + _match(s[i - 1], t[j - 1])
            left = m[i - 1, j] + gap
            up = m[i, j - 1] + gap
            m[i][j] = max(0, left, up, diag)
            if m[i][j] == 0:
                pointer[i][j] = 0
            if m[i][j] == left:
                pointer[i][j] = 1
            if m[i][j] == up:
                pointer[i][j] = 2
            if m[i][j] == diag:
                pointer[i][j] = 3
            if m[i][j] >= max_m:
                max_i = i
                max_j = j
                max_m = m[i][j];

    align1, align2 = '', ''

    i, j = max_i, max_j

    # Traceback
    while pointer[i][j] != 0:
        if pointer[i][j] == 3:
            align1 += s[i - 1]
            align2 += t[j - 1]
            i -= 1
            j -= 1
        elif pointer[i][j] == 2:
            align1 += '-'
            align2 += t[j - 1]
            j -= 1
        elif pointer[i][j] == 1:
            align1 += s[i - 1]
            align2 += '-'
            i -= 1

    print(align1[::-1])
    print(align2[::-1])

s, t = 'tccCAGTTATGTCAGgggacacgagcatgcagagac', 'aattgccgccgtcgttttcagCAGTTATGTCAGatc'
s = s.upper()
t = t.upper()
dim_i, dim_j = len(s), len(t)

water(s, t)
