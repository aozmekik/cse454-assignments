import csv
from collections import defaultdict
from treelib import Node, Tree


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


def insert_tree(tree, P, node_link=None, T=None, node='null'):
    T = T if T else tree
    p, P = P[0], P[1:]
    N = [child for child in T.children(get_node(T, node)) if child.tag == p]
    N = N[0] if N else None

    if N:
        N.data += 1
    else:
        N = tree.create_node(tag=p, parent=get_node(T, node), data=1)
    node = N.tag
    node_link[node].add(N.identifier)

    if P:
        insert_tree(tree, P, node_link=node_link, T=tree.subtree(N.identifier), node=node)


def construct_tree(D, min_sup=1):
    # get support count dict and list of frequent items
    F, L = sup_count(D, min_sup)
    node_link = defaultdict(lambda: set())

    # create tree and the root node
    tree = Tree()
    tree.create_node('null', 'null')

    for trans in D:
        trans = sort(trans, L)
        insert_tree(tree, P=trans, node_link=node_link, T=tree)

    tree.show()

def fp_growth(tree, alfa):
    pass


D = read_dataset('dataset/data.csv')

construct_tree(D)


# create tree and the root node
# tree = Tree()
# node_link = defaultdict(lambda: set())
# tree.create_node('null', 'null')

# insert_tree(tree, ['jane', 'semih'], node_link=node_link)
# tree.show()

# insert_tree(tree, ['jane', 'semih'], node_link=node_link)
# tree.show()

# insert_tree(tree, ['semih', 'jane'], node_link=node_link)

# tree.show()
# print(node_link)