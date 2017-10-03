import numpy as np

start_gap = 1
continue_gap = -0.5
match = 1.
mismatch = -1
MIN = -float("inf")


def _match(s, t, i, j):
    if t[i - 1] == s[j - 1]:
        return match
    else:
        return mismatch


# Initialization
def init_lower_level(dim_i, dim_j):
    lower_level = np.zeros((dim_i, dim_j))
    for i in range(dim_i):
        for j in range(dim_j):
            if i > 0 and j == 0:
                lower_level[i, j] = start_gap + (continue_gap * i)
            elif j > 0:
                lower_level[i, j] = MIN
    return lower_level


def init_upper_level(dim_i, dim_j):
    upper_level = np.zeros((dim_i, dim_j))
    for i in range(dim_i):
        for j in range(dim_j):
            if j > 0 and i == 0:
                upper_level[i, j] = start_gap + (continue_gap * j)
            elif i > 0:
                upper_level[i, j] = MIN
    return upper_level


def init_main_level(dim_i, dim_j):
    main_level = np.zeros((dim_i, dim_j))
    for i in range(dim_i):
        for j in range(dim_j):
            if j == 0 and i != 0 or j != 0 and i == 0:
                main_level[i, j] = MIN
    return main_level

s, t = 'tccCAGTTATGTC', 'aattgccgc'
s = s.upper()
t = t.upper()
dim_i = len(t) + 1
dim_j = len(s) + 1

#build matrix
lower_level = init_lower_level(dim_i, dim_j)
upper_level = init_upper_level(dim_i, dim_j)
main_level = init_main_level(dim_i, dim_j)

print(main_level)
print(upper_level)
print(lower_level)
gap = False
for j in range(1, dim_j):
    for i in range(1, dim_i):
        lower_level[i, j] = max((start_gap + continue_gap + main_level[i, j - 1]), (continue_gap + lower_level[i, j - 1]))

        upper_level[i, j] = max((start_gap + continue_gap + main_level[i - 1, j]), (continue_gap + upper_level[i - 1, j]))

        main_level[i, j] = max(_match(s, t, i, j) + main_level[i - 1, j - 1], lower_level[i, j], upper_level[i, j])
        print('STEP', j, i)
        print(main_level)
        print(upper_level)
        print(lower_level)
# backtrace
align1 = ''
align2 = ''
i = len(t)
j = len(s)
while i > 0 or j > 0:
    if i > 0 and j > 0 and main_level[i, j] == main_level[i - 1, j - 1] + _match(s, t, i, j):
        align1 += s[j - 1]
        align2 += t[i - 1]
        i -= 1
        j -= 1
    elif i > 0 and main_level[i, j] == upper_level[i, j] or j == 0 and main_level[i, j] == start_gap:
        align1 += '_'
        align2 += t[i - 1]
        i -= 1
    elif j > 0 and main_level[i, j] == lower_level[i, j] or i == 0 and main_level[i, j] == start_gap:
        align1 += s[j - 1]
        align2 += '_'
        j -= 1

str1 = ' '.join([align1[j] for j in range(-1, -(len(align1) + 1), -1)])
str2 = ' '.join([align2[j] for j in range(-1, -(len(align2) + 1), -1)])

print(str1)
print(str2)
print("\n")
