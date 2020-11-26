import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import random
from scipy.spatial import distance
import copy
import seaborn as sns


def distance_matrix(X):
    return distance.squareform(distance.pdist(X))


def neighbours(X, D, i, eps):
    N = []
    for i, dist in enumerate(D[i]):
        if dist <= eps:
            N.append(i)
    return set(N)


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

X[:, -1] = label

# rand_state = 11
# color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list(
#     '', ['red', 'cyan', 'magenta', 'blue', 'gray', 'yellow'])
# cdict = {-1: 'red', -2: 'blue', 0: 'green', 1: 'yellow',
#          2: 'cyan', 3: 'magenta', 4: 'black', 5: 'orange', 6: 'gray',7: }

# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
# plt_ind_list = np.arange(3)+131s

# # for std, plt_ind in zip([0.5, 1, 10], plt_ind_list):
# #     # x, label = dt.make_blobs(n_features=2,
# #     #                          centers=4,
# #     #                          cluster_std=std,
# #     #                          random_state=rand_state)

# #     plt.subplot(plt_ind)
# #     my_scatter_plot = plt.scatter(x[:, 0],
# #                                   x[:, 1],
# #                                   c=label,
# #                                   vmin=min(label),
# #                                   vmax=max(label),
# #                                   cmap=color_map_discrete)
# #     plt.title('cluster_std: ' + str(std))

# x, label = dt.make_moons(n_samples=1500, noise=.1)
# plt.subplot(plt_ind_list[0])

# fig, ax = plt.subplots()
# for l in np.unique(label):
#     ix = np.where(label == l)
#     ax.scatter(X[:, 0], X[:, 1], label=l,
#                vmin=min(label), vmax=max(label),)

# ax.legend()
# my_scatter_plot = plt.scatter(X[:, 0],
#                               X[:, 1],
#                               c=label,
#                               vmin=min(label),
#                               vmax=max(label),
#                               color="smoker")
# # plt.title('cluster_std')
# plt.colorbar()

# fig.subplots_adjust(hspace=0.3, wspace=.3)

groups = X.groupby("Category")

plt.suptitle('clusters', fontsize=20)
plt.show()
