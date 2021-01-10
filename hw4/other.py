from sklearn.datasets import load_iris, load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
import math
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# TODO.
# implement filter feature selection
# implement wrapper feature selection
# pca tool
# lda tool


def f1_score(true, predict):
    true = set([(k, v) for k, v in enumerate(true)])
    predict = set([(k, v) for k, v in enumerate(predict)])
    true = set(true)
    predict = set(predict)
    tp = len(true & predict)
    fp = len(predict) - tp
    fn = len(true) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (precision * recall) / (precision + recall)


def accuracy_score(true, predict):
    print('accuracy: ', np.sum(true == predict) / true.shape[0])


def g(x, mean, std):
    '''
        probability density function.
    '''
    return 1/(math.sqrt(2*math.pi)*std) * math.pow(math.e, -((x-mean)**2)/(2*(std**2)))


def test(X_train, Y_train, X_val, Y_val, categorical=False):
    from sklearn.naive_bayes import GaussianNB, CategoricalNB
    if categorical:
        model = CategoricalNB()
    else:
        model = GaussianNB()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_val)
    # accuracy_score(Y_val, Y_pred)
    return f1_score(Y_val, Y_pred)


def nb(X_train, Y_train, X_test, categorical=False):
    if categorical:
        # count dist.
        df = X_train.groupby(Y_train)
        count = [df[c].value_counts() for c in X_train.columns]

        distinct_cols = [len(X_train[col].unique()) for col in X_train.columns]
    else:
        # collect means and standart deviations for gaussian distribution.
        means = X_train.groupby(Y_train).apply(np.mean)
        stds = X_train.groupby(Y_train).apply(np.std)

    # class distribution
    PCi = X_train.groupby(Y_train).apply(lambda x: len(x)) / X_train.shape[0]

    Y_pred = []
    for i in range(X_test.shape[0]):  # row iterate
        p = {}

        for Ci in np.unique(Y_train):  # class iterate
            p[Ci] = PCi.iloc[Ci]
            for j, val in enumerate(X_test.iloc[i]):  # column iterate
                if categorical:
                    # applying laplace smooth 1.
                    V = count[j][Ci].sum() + distinct_cols[j]
                    p[Ci] *= ((count[j][Ci, val] + 1)
                              if (Ci, val) in count[j] else 1) / V
                else:
                    p[Ci] *= g(val, means.iloc[Ci, j], stds.iloc[Ci, j])

        Y_pred.append(pd.Series(p).values.argmax())
    return Y_pred


def demo_numerical(pca=False, lda=False):
    data = load_breast_cancer()

    X, Y, column_names = data['data'], data['target'], data['feature_names']
    X = pd.DataFrame(X, columns=column_names)

    if pca:
        print(X.shape)
        pca = PCA(n_components=4)
        X = pd.DataFrame(data=pca.fit_transform(X))
        print(X.shape)
    if lda:
        print(X.shape)
        lda = LinearDiscriminantAnalysis()
        X = pd.DataFrame(data=lda.fit_transform(X, Y))
        print(X.shape)


    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    my_score = []
    true_score = []
    for train_index, test_index in cv.split(X):
        X_train, X_test, Y_train, Y_test = X.iloc[train_index], X.iloc[test_index], Y[train_index], Y[test_index]

        Y_pred = nb(X_train, Y_train, X_test)
        my_score.append(f1_score(Y_test, Y_pred))
        true_score.append(test(X_train, Y_train, X_test, Y_test))

    print(np.mean(my_score))
    print(np.mean(true_score))


def demo_categorical():
    df = pd.read_csv('data/tennis.csv')

    for col in df.columns:
        df[col] = df[col].astype('category').cat.codes

    X = df.drop('play', axis=1)
    Y = df['play']
    Y = np.array(Y)

    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    my_score = []
    true_score = []
    for train_index, test_index in cv.split(X):
        X_train, X_test, Y_train, Y_test = X.iloc[train_index], X.iloc[test_index], Y[train_index], Y[test_index]

        Y_pred = nb(X_train, Y_train, X_test, categorical=True)
        my_score.append(f1_score(Y_test, Y_pred))
        true_score.append(
            test(X_train, Y_train, X_test, Y_test, categorical=True))

    print(np.mean(my_score))
    print(np.mean(true_score))



# demo_numerical()
# demo_categorical()

demo_numerical()
demo_numerical(pca=True)
demo_numerical(lda=True)
