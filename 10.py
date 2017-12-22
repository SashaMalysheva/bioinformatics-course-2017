import numpy as np
import matplotlib.cm as cm
import random
import matplotlib.pyplot as plt
import math

random.seed(1003)

class Tree:
    def __init__(self, p=None, left=None, right=None, dist_l=0, dist_r=0, name=None):
        self.dist_l = dist_l
        self.dist_r = dist_r
        self.points = p
        self.right = right
        self.left = left
        self.name = name

    def to_str(self, s=""):
        if self.name != None:
            s += str(self.name)
        if self.left != None:
            s += "{" + self.left.to_str() + ":" + str(self.dist_l) + ', '
        if self.right != None:
            s += self.right.to_str() + ":" + str(self.dist_r) + '}'
        return s


def update_matrix_d(d, i, j):
    new_d = np.copy(d)
    new_d=np.delete(new_d, [i, j], axis=0)
    new_d=np.delete(new_d, [i, j], axis=1)
    v = np.delete((d[i] + d[j])/2, [i, j])
    v -= d[i,j]/2
    v = v.reshape((v.shape[0], 1))
    new_d = np.append(new_d, v, 1)
    v = np.append(v, 0)
    v = v.reshape((1, v.shape[0]))
    new_d = np.append(new_d, v, 0)
    return new_d

def wpgma(d, names):
    # Initialize
    nodes = []
    n = d.shape[0]
    for i in range(n):
        node = Tree(p=[i], name=names[i])
        nodes = nodes + [node]
    # Iterate until the number of clusters is k
    while n > 2:
        c1, c2 = 0, 0
        dist_l, dist_r = 0, 0
        num1, num2 = 0, 0
        sdis = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                m_i = (np.sum(d[i,:]) - d[i,i] - d[i,j])/(n - 2)
                m_j = (np.sum(d[j,:]) - d[j,j] - d[i,j])/(n - 2)
                if d[i, j] - m_i - m_j < sdis:
                    sdis = d[i, j] - m_i - m_j
                    c1, c2 = nodes[i], nodes[j]
                    num1, num2 = i, j
                    dist_l = 0.5 * (d[i,j] + m_i - m_j)
                    dist_r = 0.5 * (d[i,j] - m_i + m_j)

        node = Tree(p=c1.points + c2.points, left=c1, right=c2, dist_l=dist_l, dist_r=dist_r,
                    name=str(nodes[num1].name) + str(nodes[num2].name))
        nodes.pop(max(num2, num1))
        nodes.pop(min(num2, num1))
        nodes = nodes + [node]
        d = update_matrix_d(d, num1, num2)
        n-=1
    return print(nodes[0].to_str(), nodes[1].to_str())



m = np.array([[0,0,0,0,0,0],[5,0,0,0,0,0],[4,7,0,0,0,0],[7,10,7,0,0,0],[6,9,6,5,0,0], [8,11,8,9,8,0]])
n=6
for i in range(n):
    for j in range(0, i):
        m[j,i] = m[i,j]
names = ['A', 'B', 'C', 'D', 'E', 'F']
node = wpgma(m, names)


