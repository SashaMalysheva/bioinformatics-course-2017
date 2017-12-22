import numpy as np
import matplotlib.cm as cm
import random
import matplotlib.pyplot as plt
import math

random.seed(1003)
INF = 1000000

class Tree:
    def __init__(self, p=None, left=None, right=None, dist=0, name=None):
        self.dist = dist
        self.points = p
        self.right = right
        self.left = left
        self.name = name

    def to_str(self, s=""):
        if self.name != None:
            s += str(self.name)
        if self.left != None:
            s += "{" + self.left.to_str() + ":" + str(self.dist - self.left.dist) + ','
        if self.right != None:
            s += self.right.to_str() + ":" + str(self.dist - self.right.dist) + '}'
        return s

def to_dist_matrix(points):
    """ Convert a set of points into a distance matrix based on a certain distance measure """
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = math.sqrt(np.sum((points[i]-points[j])**2))
    return dist

def euclidistance(c1, c2):
    dist = .0
    n1 = len(c1.points)
    n2 = len(c2.points)
    for i in range(n1):
        for j in range(n2):
            dist += math.sqrt(np.sum(c1.points[i] - c2.points[j]) ** 2)
    dist = dist / (n1 * n2)
    return dist

def update_matrix(m, i, j, wi=1, wj=1):
    new_m = np.copy(m)
    new_m=np.delete(new_m, [i, j], axis=0)
    new_m=np.delete(new_m, [i, j], axis=1)
    v = np.delete((m[i]*wi + m[j]*wj)/(wi + wj), [i, j])
    v = v.reshape((v.shape[0], 1))
    new_m = np.append(new_m, v, 1)
    v = np.append(v, 0)
    v = v.reshape((1, v.shape[0]))
    new_m = np.append(new_m, v, 0)
    return new_m

def wpgma(m, names):
    # Initialize
    nodes = []
    n = m.shape[0]
    for i in range(n):
        node = Tree(p=[i], name=names[i])
        nodes = nodes + [node]
    # Iterate until the number of clusters is k
    while n > 1:
        c1, c2 = 0, 0
        num1, num2 = 0, 0
        sdis = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                if m[i, j] < sdis:
                    sdis = m[i, j]
                    c1, c2 = nodes[i], nodes[j]
                    num1, num2 = i, j
        node = Tree(p=c1.points + c2.points, left=c1, right=c2, dist=m[num1, num2]/2,
                    name=str(nodes[num1].name) + str(nodes[num2].name))
        nodes.pop(max(num2, num1))
        nodes.pop(min(num2, num1))
        nodes = nodes + [node]
        m = update_matrix(m, num1, num2)
        n-=1
    return nodes[0]

def upgma(m, names):
    # Initialize
    nodes = []
    n = m.shape[0]
    for i in range(n):
        node = Tree(p=[i], name=names[i])
        nodes = nodes + [node]
    # Iterate until the number of clusters is k
    while n > 1:
        c1, c2 = 0, 0
        num1, num2 = 0, 0
        sdis = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                if m[i, j] < sdis:
                    sdis = m[i, j]
                    c1, c2 = nodes[i], nodes[j]
                    num1, num2 = i, j

        m = update_matrix(m, num1, num2, len(nodes[num1].points), len(nodes[num2].points))
        node = Tree(p=c1.points + c2.points, left=c1, right=c2, dist=sdis/2,
                    name=str(nodes[num1].name) + str(nodes[num2].name))
        nodes.pop(max(num2, num1))
        nodes.pop(min(num2, num1))
        nodes = nodes + [node]
        n-=1
    return nodes[0]

def upgma_2d(points, names):
    plt.ion()
    fig = plt.figure()
    # Initialize
    nodes = []
    n = len(points)
    for i in range(n):
        node = Tree(p=[points[i]], name=names[i])
        nodes = nodes + [node]
    # Iterate until the number of clusters is k
    while n > 1:
        c1, c2 = 0, 0
        num1, num2 = 0, 0
        sdis = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                d = euclidistance(nodes[i], nodes[j])
                if d < sdis:
                    sdis = d
                    c1, c2 = nodes[i], nodes[j]
                    num1, num2 = i, j
        # Remove
        node = Tree(p=c1.points + c2.points, left=c1, right=c2, dist=sdis/2,
                    name=str(nodes[num1].name) + str(nodes[num2].name))

        nodes.pop(max(num2, num1))
        nodes.pop(min(num2, num1))
        nodes = nodes + [node]
        n-=1

        # Plot
        colors = cm.rainbow(np.linspace(0, 1, n))
        for i, c in zip(range(n), colors):
            plt.plot([x[0] for x in nodes[i].points], [x[1] for x in nodes[i].points], color=c, marker="^")
        plt.show()
        plt.pause(0.1)
    plt.show()
    plt.pause(5)
    return nodes[0]

m = np.array([[0,16,16,10],[16,0,8,8],[16,8,0,4],[10,8,4,0]])
for i in range(m.shape[0]):
    m[i,i] = INF
names = ['K', 'L', 'M', 'N']


m = np.array([[0,0,0,0,0,0],[5,0,0,0,0,0],[4,7,0,0,0,0],[7,10,7,0,0,0],[6,9,6,5,0,0], [8,11,8,9,8,0]])
n=6
for i in range(n):
    for j in range(0, i):
        m[j,i] = m[i,j]
names = ['A', 'B', 'C', 'D', 'E', 'F']



node = wpgma(m, names)
print(node.to_str())
node = upgma(m, names)
print(node.to_str())

datapoints = [(random.normalvariate(2.5, 1.0), random.normalvariate(1.5,1.0)) for i in range(20)] + \
				[(random.normalvariate(-1, 0.5), random.normalvariate(3,0.5)) for i in range(20)]
names = np.arange(len(datapoints))
datapoints = np.array(datapoints)
# Normalize the data
datapoints[:, 0] -= np.sum(datapoints[:, 0])
datapoints[:, 1] -= np.sum(datapoints[:, 1])
datapoints[:, 0] /= math.sqrt(np.sum(datapoints[:, 0] ** 2))
datapoints[:, 1] /= math.sqrt(np.sum(datapoints[:, 1] ** 2))

node = upgma_2d(datapoints, names)
print(node.to_str())

m = np.array([[0,0,0,0,0,0],[5,0,0,0,0,0],[4,7,0,0,0,0],[7,10,7,0,0,0],[6,9,6,5,0,0], [8,11,8,9,8,0]])
n=6
for i in range(n):
    for j in range(0, i):
        m[j,i] = m[i,j]
names = ['A', 'B', 'C', 'D', 'E', 'F']

