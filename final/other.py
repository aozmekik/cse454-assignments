import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import numpy as np
import pickle
from test import *
from cfs import CFS

target_names = ['ekonomi', 'kultursanat',
                'saglik', 'siyaset', 'spor', 'teknoloji']


def save_dataset(preprocess, save=None):
    dataset = 'TTC-3600/TTC-3600_Orj'

    X, y = [],  []
    for root, directories, files in os.walk(dataset):
        for directory in directories:
            for parent, _, files in os.walk(dataset + '/' + directory):
                y += [directories.index(directory)] * len(files)
                i = 0
                for file in files:
                    with open(parent + '/' + file) as f:
                        X.append(preprocess(f.read()))
                    print('{}/{}'.format(len(X), 3600))

    y = np.array(y)

    if save:
        with open(save, 'wb') as file:
            pickle.dump((X, y), file)

    return X, y


def load_dataset(ds):
    objects = []
    with (open('processed/' + ds, 'rb')) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects[0]


sns.set()  # use seaborn plotting style


class Classifier:
    __RANDOM_STATE = 42

    def __init__(self, dataset, method, max_features=None, fs=None):
        self.X, self.y = load_dataset(dataset)
        self.X = [strip_numbers(x) for x in self.X]

        tfidfconverter = TfidfVectorizer(max_features=5000)
        self.X = tfidfconverter.fit_transform(self.X).toarray()

        if method == 'NB':
            self.model = MultinomialNB()
        elif method == 'RF':
            self.model = RandomForestClassifier(
                max_depth=128, random_state=self.__RANDOM_STATE)
        elif method == 'SVM':
            self.model = SVC(C=1.0, kernel='linear', degree=3,
                             gamma='auto', cache_size=7000)
        elif method == 'KNN':
            self.model = KNeighborsClassifier(n_neighbors=5)
        elif method == 'J48':
            self.model = DecisionTreeClassifier()

        if fs == 'cfs':
            idx = CFS.cfs(self.X, self.y)
            print(idx)

    def fit(self):
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        score = []
        for train_index, test_index in cv.split(self.X):
            X_train, X_test, y_train, y_test = self.X[train_index], self.X[
                test_index], self.y[train_index], self.y[test_index]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            score.append(accuracy_score(y_test, y_pred))
        print(np.mean(score))

    def cfmatrix(self, y_test, y_pred):
        # plot the confusion matrix
        mat = confusion_matrix(y_test, y_pred)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d',
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('true labels')
        plt.ylabel('predicted label')
        plt.show()


# methods = ['J48']

cf = Classifier(dataset='zembds_stopword', method='NB', fs='cfs')
cf.fit()

# CFS.cfs(X, y)

# rf(X, y)
# j48(X, y)
