import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import random
from scipy.spatial import distance
import copy
import seaborn as sns
import plotly.express as px


def distance_matrix(X):
    return distance.squareform(distance.pdist(X))

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def neighbours(X, D, i, eps):
    N = []
    for i, dist in enumerate(D[i]):
        if dist <= eps:
            N.append(i)
    return set(N)

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def db_scan(X, min_pts=0, eps=0):

    D = distance_matrix(X)

    unvisited = -1
    noise = -2
    label = np.array([unvisited for i in range(len(X))])

    db = [i for i in range(len(X))]
    random.shuffle(db)

    Ci = -1
    for p in db:
        if label[p] != unvisited:
            continue
        N = neighbours(X, D, p, eps)
        if len(N) < min_pts:
            label[p] = noise
            continue

        Ci += 1
        label[p] = Ci
        S = copy.deepcopy(N)   # N \ {p}
        S.remove(p)

        for q in S:
            if label[q] == noise:
                label[q] = Ci
            if label[q] != unvisited:
                continue
            label[q] = Ci
            N = neighbours(X, D, q, eps)
            if len(N) >= min_pts:
                S.union(N)

    return label


X, Y = dt.make_moons(n_samples=100, noise=.1)
label = db_scan(X, min_pts=3, eps=0.5)

label = [l + 1 for l in label]

# FIXME. color palette.
# TODO. on real data.
# 

fig, ax = plt.subplots()
fig.set_size_inches(10.5, 6.5, forward=True)
print(len(set(label)))
scatter = ax.scatter(X[:, 0], X[:, 1], c=label, cmap=discrete_cmap(len(set(label)), 'cubehelix'))


# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    title="Clusters", borderaxespad=0, bbox_to_anchor=(1.04, 1), loc='upper left')
ax.add_artist(legend1)

# produce a legend with a cross section of sizes from the scatter
handles, labels = scatter.legend_elements(prop="sizes", alpha=1)

plt.show()

# X[:, -1] = label

# sns.lmplot('carat', 'price', data=X, hue='color', fit_reg=False)
# plt.suptitle('clusters', fontsize=10)
# plt.show()
