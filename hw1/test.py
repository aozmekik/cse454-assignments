import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

import sklearn.datasets as dt
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import random
from scipy.spatial import distance
import copy
import seaborn as sns
import plotly.express as px


# TODO. find a real dataset with two dims
# TODO. prepare a report.


def distance_matrix(X):
    return distance.squareform(distance.pdist(X))


def discrete_cmap(N):
    """Create an N-bin discrete colormap for N discrete items"""

    # lineer color palette generated for N labels.
    base = plt.cm.gist_ncar
    color_list = base(np.linspace(0, 0.9, N))

    # pushing similar colors apart
    N = len(color_list)
    for i in range(0, int(N/2)):
        if i % 2 == 0:
            j = N-i-1
            color_list[j], color_list[i] = copy.deepcopy(
                color_list[i]), copy.deepcopy(color_list[j])

    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def neighbours(X, D, i, eps):
    N = []
    for i, dist in enumerate(D[i]):
        if dist < eps:
            N.append(i)
    return N

# def neighbours(X, i, eps):
#     N = []
#     for p in range(0, len(X)):
#         if np.linalg.norm(X[i] - X[p]) < eps:
#             N.append(p)
#     return N


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


# X, Y = dt.make_moons(n_samples=100, noise=.1)
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)

Y = db_scan(X, eps=0.1, min_pts=2)

test(X, Y, 0.1, 2)

# fig, ax = plt.subplots()
# fig.set_size_inches(10.5, 6.5, forward=True)

# scatter = ax.scatter(X[:, 0], X[:, 1], c=Y,
#                      cmap=discrete_cmap(len(set(Y))))


# # produce a legend with the unique colors from the scatter
# legend1 = ax.legend(*scatter.legend_elements(),
#                     title="Clusters", borderaxespad=0, bbox_to_anchor=(1.04, 1), loc='upper left')
# ax.add_artist(legend1)

# # produce a legend with a cross section of sizes from the scatter
# handles, labels = scatter.legend_elements(prop="sizes", alpha=1)

# plt.show()
