import itertools
from treelib import Node, Tree
from collections import defaultdict
import csv
import warnings

# for to test our implementation
import pyfpgrowth



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


def sup_count(D):
    # set of frequent items and their support counts
    F = defaultdict(lambda: 0)

    for trans in D:
        for item in trans:
            F[item] += 1

    # filter by support threshold
    # F = dict(filter(lambda x: x[1] >= min_sup, dict(F).items()))

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
    header[node].append(N.identifier)

    if P:
        insert_tree(tree, P, header=header,
                    T=tree.subtree(N.identifier), node=node)


def construct_tree(D):
    # get support count dict and list of frequent items
    F, L = sup_count(D)
    header = defaultdict(lambda: list())

    # create tree and the root node
    tree = Tree()
    tree.create_node('null', 'null')

    for trans in D:
        trans = sort(trans, L)
        insert_tree(tree, P=trans, header=header, T=tree)

    for k, v in header.items():
        res = []
        for i in header[k]:
            if i not in res:
                res.append(i)
        header[k] = res

    # tree.show()
    return tree, header, F, L


def combinations(lst):
    C = []
    for r in range(0, len(lst) + 1):
        for subset in itertools.combinations(lst, r):
            C.append(subset)
    return C


def cpb(tree, item):
    base = {}
    D = []
    node = tree.get_node(item)
    backtrace = ()
    while node.tag != 'null':
        node = tree.get_node(node.predecessor(tree.identifier))
        if node.tag != 'null':
            backtrace += (node.tag, )
    sup_count = tree.get_node(item).data

    # reverse trace because the order starts from root
    backtrace = backtrace[:: -1]

    if backtrace:
        base[backtrace] = sup_count
    for _ in range(sup_count):
        if backtrace:
            D.append(backtrace)
    return base, D


def clean_cpb(D, min_sup):
    D = [('i2', 'i1'), ('i2', )]
    item = defaultdict(lambda: 0)
    for d in D:
        for i in d:
            item[i] += 1

    new_D = []
    for d in D:
        m = ()
        for i in d:
            if item[i] >= min_sup:
                m += (i, )
        new_D.append(m)
    return new_D


def fp_growth(tree, header, F, L, alfa=None, min_sup=2):
    pattern = []

    paths = tree.paths_to_leaves()
    # tree contains a single path
    if len(paths) == 1:
        P = paths[0][1:]
        for beta in combinations(P):
            if beta:
                sup = min([tree.get_node(b).data for b in beta])
                beta = list(map(lambda x: tree.get_node(x).tag, beta))
                # if sup >= min_sup:
                if alfa:
                    pattern.append({tuple(beta + alfa): sup})
                else:
                    pattern.append({tuple(beta): sup})
    else:
        for ai in list(header.keys())[::-1]:
            for node in header[ai]:
                sup = tree.get_node(node).data
                beta = [ai] + alfa if alfa else [ai]
                # if alfa:
                pattern.append({tuple(beta): sup})
                base, D = cpb(tree, node)
                if base:
                    beta_tree, beta_header, beta_F, beta_L = construct_tree(D)
                else:
                    beta_tree = None
                if beta_tree and len(beta_tree) != 0:
                    pattern += fp_growth(beta_tree,
                                         beta_header, beta_F, beta_L, beta)
    return pattern


def fpg(filename, min_sup=2, verbose=False):
    D = read_dataset(filename)

    tree, header, F, L = construct_tree(D)
    pattern = fp_growth(tree, header, F, L)

    P = defaultdict(lambda: 0)
    for p in pattern:
        for k, v in p.items():
            P[k] += v
    P = {k: v for k, v in dict(P).items() if v >= min_sup}

    if verbose:
        for k, v in P.items():
            print(k, ':', v)
    return P

def test(fname):
    D = []
    # collect transactions from file
    with open(fname, 'r') as file:
        for trans in csv.reader(file):
            D.append(list(trans))

    patterns = pyfpgrowth.find_frequent_patterns(D, 2)
    # for k, v in patterns.items():
    #     print(k, ':', v)

    ours = fpg(fname)
    patterns = {tuple(sorted(k)): v for k, v in patterns.items()}
    ours = {tuple(sorted(k)): v for k, v in ours.items()}

    for k, v in patterns.items():
        if k not in ours or v != ours[k]:
            print('Fail!')
            break
    print('All matches!')


test('dataset/tesco2.csv')

fpg('dataset/data0.csv', verbose=True)