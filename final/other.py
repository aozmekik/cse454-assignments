import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pickle
from test import *

target_names = ['ekonomi', 'kultursanat',
                    'saglik', 'siyaset', 'spor', 'teknoloji']
def get_dataset(preprocess, save=None):
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


def read_ds(ds):
    objects = []
    with (open("processed/" + ds, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects[0]


# get_dataset(preprocess=preprocess, save='originalds')
# get_dataset(preprocess=lambda x: preprocess(x, stemming='fps5'), save='f5ds')
# get_dataset(preprocess=lambda x: preprocess(x, stemming='fps7'), save='f7ds')
# get_dataset(preprocess=lambda x: preprocess(x, stemming='zemb'), save='zembds')


# get_dataset(preprocess=lambda x: preprocess(
#     x, stopword=True), save='originalds_stopword')
# get_dataset(preprocess=lambda x: preprocess(
#     x, stemming='fps5', stopword=True), save='f5ds_stopword')
# get_dataset(preprocess=lambda x: preprocess(
#     x, stemming='fps7', stopword=True), save='f7ds_stopword')
# get_dataset(preprocess=lambda x: preprocess(
#     x, stemming='zemb', stopword=True), save='zembds_stopword')

X, y = read_ds('f7ds_stopword')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

sns.set()  # use seaborn plotting style

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# Train the model using the training data
model.fit(X_train, y_train)
# Predict the categories of the test data
predicted_categories = model.predict(X_test)


# plot the confusion matrix
mat = confusion_matrix(y_test, predicted_categories)
sns.heatmap(mat.T, square=True, annot=True, fmt="d",
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()


print("The accuracy is {}".format(accuracy_score(y_test, predicted_categories)))
