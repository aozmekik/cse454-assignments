from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

def accuracy_score(a, b):
    '''
        returns the percentage match score between the two sets.
    '''
    return np.sum(a == b) / a.shape[0]

def g(x, mean, std):
    '''
        probability density function.
    '''
    return 1/(math.sqrt(2*math.pi)*std) * math.pow(math.e, -((x-mean)**2)/(2*(std**2)))


def test(X_train, Y_train, X_val, Y_val):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, Y_train)
    print(accuracy_score(Y_val, model.predict(X_val)))

def cat(X_train, Y_train, X_val, Y_val):
    # class distribution
    PCi = X_train.groupby(Y_train).apply(lambda x: len(x)) / X_train.shape[0]

    Y_pred = []
    for i in range(X_val.shape[0]):
        p = {}

        for Ci in np.unique(Y_train):
            p[Ci] = PCi.iloc[Ci]
            for j, val in enumerate(X_val.iloc[i]):
                # c sınıfına ait k row'unda x değerlerini alanların sayısı / c sınıfına ait tüm tuple ler.
                p[Ci] *= X_train.groupby([j, Ci]).size() / X_train.groupby([Ci]).size()


def con(X_train, Y_train, X_val, Y_val):
    # collect means and standart deviations for gaussian distribution.
    means = X_train.groupby(Y_train).apply(np.mean)
    stds = X_train.groupby(Y_train).apply(np.std)

    # class distribution
    PCi = X_train.groupby(Y_train).apply(lambda x: len(x)) / X_train.shape[0]

    Y_pred = []
    for i in range(X_val.shape[0]):
        p = {}

        for Ci in np.unique(Y_train):
            p[Ci] = PCi.iloc[Ci]
            for j, val in enumerate(X_val.iloc[i]):
                p[Ci] *= g(val, means.iloc[Ci, j], stds.iloc[Ci, j])
                
        Y_pred.append(pd.Series(p).values.argmax())
    print(accuracy_score(Y_val, Y_pred))
    

data = load_iris()


a, Y, column_names = data['data'], data['target'], data['feature_names']
a = pd.DataFrame(a, columns=column_names)


X_train, X_val, Y_train, Y_val = train_test_split(a, Y, random_state=44)

con(X_train, Y_train, X_val, Y_val)

test(X_train, Y_train, X_val, Y_val)