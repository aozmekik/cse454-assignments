import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

import sklearn.datasets as dt
from sklearn.cluster import DBSCAN

import random
from scipy.spatial import distance
import copy


def distance_matrix(X):
    return distance.squareform(distance.pdist(X))

def neighbours(X, D, i, eps):
    N = []
    for i, dist in enumerate(D[i]):
        if dist < eps:
            N.append(i)
    return N

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def db_scan(X, eps=0, min_pts=0):

    D = distance_matrix(X)

    unvisited = 0
    noise = -1
    labels = np.array([unvisited for i in range(len(X))])

    db = [i for i in range(len(X))]
    # random.shuffle(db)

    Ci = 0
    for p in db:
        if labels[p] != unvisited:
            continue
        N = neighbours(X, D, p, eps)
        if len(N) < min_pts:
            labels[p] = noise
            continue

        Ci += 1
        labels[p] = Ci

        # N \ {p}
        S = copy.deepcopy(N)
        i = 0
        while i < len(S):
            q = S[i]
            if labels[q] == noise:
                labels[q] = Ci
            elif labels[q] == unvisited:
                labels[q] = Ci
                N = neighbours(X, D, q, eps)
                if len(N) >= min_pts:
                    S += N
            i += 1

    return labels


def test(X, Y, eps, min_pts):
    db = DBSCAN(eps=eps, min_samples=min_pts).fit(X)
    labels = db.labels_

    fail = 0

    for i in range(0, len(labels)):
        if not labels[i] == -1:
            labels[i] += 1

    # compare each label
    for i in range(0, len(labels)):
        if not labels[i] == Y[i]:
            print('Scikit learn:', labels[i], 'mine:', Y[i])
            fail += 1

    if fail == 0:
        print('PASS - All labels match!')
    else:
        print('FAIL -', fail, 'labels don\'t match.')


n_samples = 750
# X, _ = dt.make_moons(n_samples=n_samples, noise=.1)
# X, _ = dt.make_circles(n_samples=n_samples, factor=.5, noise=.05)

centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
X, _ = dt.make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.4,
                            random_state=0)


Y = db_scan(X, eps=0.28, min_pts=40)

# test(X, Y, 0.2, 10)

fig, ax = plt.subplots()
fig.set_size_inches(10.5, 6.5, forward=True)

for y in np.unique(Y):
    i = np.where(Y == y)
    ax.scatter(X[i][:, 0], X[i][:, 1], label=y)
ax.legend()
plt.show()