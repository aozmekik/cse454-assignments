from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

from collections import defaultdict

def naive_bayes(X, Y):
    PCi = defaultdict(lambda: 0)
    for y in Y:
        PCi[y] += 1
    for k, v in PCi.items():
        PCi[k] /= len(Y)

    PXCi = defaultdict(lambda: 0)
        


data = load_iris()


X, Y, column_names = data['data'], data['target'], data['feature_names']
X = pd.DataFrame(X, columns=column_names)


print(X)
print(Y)



# data = pd.read_csv('data/tennis.csv')


# X = pd.get_dummies(data[['outlook', 'temp', 'humidity', 'windy']])
# Y = pd.DataFrame(data['play'])
# print(X, Y)

# # split data
