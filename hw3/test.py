import csv
from collections import defaultdict
from treelib import Node, Tree
import itertools


def read_dataset(fname):
    '''
        reads dataset with given file name and returns the transaction list
    '''

    # transaction list
    D = []

    # collect transactions from file
    with open(fname, 'r') as file:
        for trans in csv.reader(file):
            D.append(set(trans))

    return D


def sup_count(D, min_sup=1):
    # set of frequent items and their support counts
    F = defaultdict(lambda: 0)

    for trans in D:
        for item in trans:
            F[item] += 1

    # filter by support threshold
    F = dict(filter(lambda x: x[1] >= min_sup, dict(F).items()))

    # sort F in support count descending order as L, list of frequent items
    L = [k for k, v in sorted(
        F.items(), key=lambda item: item[1], reverse=True)]
    return F, L


def sort(trans, L):
    '''
        select and sort the frequent items in trans according to the order of L.
    '''
    return [item for item in L if item in trans]


def get_node(T, tag):
    for node in T.nodes:
        node = T.get_node(node)
        if node.tag == tag:
            return node.identifier


def insert_tree(tree, P, header=None, T=None, node='null'):
    T = T if T else tree
    p, P = P[0], P[1:]
    N = [child for child in T.children(get_node(T, node)) if child.tag == p]
    N = N[0] if N else None

    if N:
        N.data += 1
    else:
        N = tree.create_node(tag=p, parent=get_node(T, node), data=1)
    node = N.tag
    header[node].add(N.identifier)

    if P:
        insert_tree(tree, P, header=header,
                    T=tree.subtree(N.identifier), node=node)


def construct_tree(D, min_sup=1):
    # get support count dict and list of frequent items
    F, L = sup_count(D, min_sup)
    header = defaultdict(lambda: set())

    # create tree and the root node
    tree = Tree()
    tree.create_node('null', 'null')

    for trans in D:
        trans = sort(trans, L)
        insert_tree(tree, P=trans, header=header, T=tree)

    tree.show()
    return tree, header, F, L

def combinations(lst):
    C = []
    for r in range(0, len(lst) + 1):
        for subset in itertools.combinations(lst, r):
            C.append(subset)
    return C

def cpb(tree, header, beta):
    base = set()
    D = []
    for item in header[beta]:
        node = tree.get_node(item)
        backtrace = set()
        while node.tag != 'null':
            node = node.parent
            backtrace.add(node.tag)
        sup_count = tree.get_node(item).data
        base[backtrace] = sup_count
        for _ in range(sup_count):
            D.append(backtrace)
    return base, D

    

# FIXME. I am not sure if it's working
# FIXME. things I am not sure of:
# 1- what to do with the support scores.
# 2- how exactly this works: what is beta?
# 3- does my cpb work and construct tree for beta also working? check.
# 4- where are we storing the patterns. 
def fp_growth(tree, header, F, L, alfa=None):
    pattern = []

    paths = tree.paths_to_leaves()
    # tree contains a single path
    if len(paths) == 1: 
        P = paths[0]
        for beta in combinations(P):
            # sup_count = min([F[tree.get_node(b).tag] for b in beta])
            pattern.append(set(beta) + set(alfa))
    else:
        for ai in header:
            beta = set(ai) + set(alfa)
            pattern.append(beta)
            base, D = cpb(tree, header, beta)
            beta_tree, beta_header, beta_F, beta_L =  construct_tree(D)
            if len(beta_tree) != 0:
                fp_growth(beta_tree, beta_header, beta_F, beta_L)
            



D = read_dataset('dataset/data.csv')
print(D)

# tree = construct_tree(D)
# fp_growth(tree)


# create tree and the root node
# tree = Tree()
# header = defaultdict(lambda: set())
# tree.create_node('null', 'null')

# insert_tree(tree, ['jane', 'semih'], header=header)
# tree.show()

# insert_tree(tree, ['jane', 'semih'], header=header)
# tree.show()

# insert_tree(tree, ['semih', 'jane'], header=header)

# tree.show()
# print(header)
