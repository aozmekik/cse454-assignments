from sklearn.feature_selection import SelectKBest, chi2

from time import time

from collections import Counter

import pandas as pd

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

import os
import numpy as np
import pickle
from test import *

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

    def __init__(self, dataset, method, max_features=8000, fs=None, n_fea='all', vector='tfidf'):
        self.dataset = dataset
        self.vector = vector
        self.X, self.y = load_dataset(dataset)
        self.X = np.array([strip_numbers(x) for x in self.X])
        self.max_features = max_features
        self.n_fea = n_fea
        self.method = method

        if method == 'NB':
            self.model = MultinomialNB()
        elif method == 'RF':
            self.model = RandomForestClassifier(
                max_depth=128, random_state=self.__RANDOM_STATE)
        elif method == 'SVM LINEAR':
            self.model = LinearSVC(cache_size=7000)
        elif method == 'SVM RBF':
            self.model = SVC(kernel='rbf', gamma=1, cache_size=7000)
        elif method == 'KNN':
            self.model = KNeighborsClassifier(n_neighbors=5)
        elif method == 'CART':
            self.model = DecisionTreeClassifier()
        elif method == 'ROCCHIO':
            self.model = NearestCentroid()

        if vector == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        elif vector == 'bagofwords':
            self.vectorizer = CountVectorizer(max_features=self.max_features)

    def fit(self):
        cv = KFold(n_splits=10, shuffle=True, random_state=0)
        score, train_time, test_time = [], [], []
        for train_index, test_index in cv.split(self.X):
            X_train, X_test, y_train, y_test = self.X[train_index], self.X[
                test_index], self.y[train_index], self.y[test_index]

            X_train = self.vectorizer.fit_transform(X_train).toarray()
            X_test = self.vectorizer.transform(X_test).toarray()

            X_train, X_test = self.select_features(
                X_train, y_train, X_test, k=self.n_fea)

            results = self.benchmark(X_train, y_train, X_test, y_test)
            score.append(results[0])
            train_time.append(results[1])
            test_time.append(results[2])

        self.print_benchmark(np.mean(score), np.mean(
            train_time), np.mean(test_time))

    def cfmatrix(self, y_test, y_pred):
        # plot the confusion matrix
        mat = confusion_matrix(y_test, y_pred)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d',
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('true labels')
        plt.ylabel('predicted label')
        plt.show()

    def select_features(self, X_train, y_train, X_test, k):
        if k == 'all':
            return X_train, X_test

        selector = SelectKBest(chi2, k=k)
        selector.fit(X_train, y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        return X_train, X_test

    def benchmark(self, X_train, y_train, X_test, y_test):
        '''
        benchmark based on f1 score
        '''
        t0 = time()
        self.model.fit(X_train, y_train)
        train_time = time() - t0

        t0 = time()
        y_pred = self.model.predict(X_test)
        test_time = time() - t0

        score = metrics.f1_score(y_test, y_pred, average='micro')
        return score, train_time, test_time

    def print_benchmark(self, score, train_time, test_time):
        print('\nmethod: ', self.method)
        print('dataset: ' + self.dataset)
        print('vector: ' + self.vector)
        print('features:\t{0}'.format(self.n_fea))
        print('train time: {0:0.4f}s'.format(train_time))
        print('test time:  {0:0.4f}s'.format(test_time))
        print('f1-score:   {0:0.4f}'.format(score))


# preprocessing features
datasets = ['originalds', 'zembds', 'f5ds', 'f7ds', 'originalds_stopword',
            'zembds_stopword', 'f5ds_stopword', 'f7ds_stopword']

# machine learning models
methods = ['RF', 'SVM LINEAR', 'SVM RBF', 'KNN', 'CART', 'ROCCHIO']

# post processing features
vectors = ['tfidf', 'bagofwords']
n_features = [500, 1000, 2000, 5000, 'all']

for dataset in datasets:
    for method in methods:
        for vector in vectors:
            for n_fea in n_features:
                cf = Classifier(dataset=dataset, method=method,
                                vector=vector, n_fea=n_fea)
                cf.fit()
