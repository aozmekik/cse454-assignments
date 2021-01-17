from sklearn.datasets import load_iris, load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
import math
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import statsmodels.api as sm


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


def demo_numerical(pca=False, process=None):
    data = load_iris()

    X, Y, column_names = data['data'], data['target'], data['feature_names']
    X = pd.DataFrame(X, columns=column_names)

    if process == "pca":
        print(X.shape)
        pca = PCA(n_components=2)
        X = pd.DataFrame(data=pca.fit_transform(X))
        print(X.shape)
    elif process == "lda":
        print(X.shape)
        lda = LinearDiscriminantAnalysis()
        X = pd.DataFrame(data=lda.fit_transform(X, Y))
        print(X.shape)
    elif process == "filter":
        X = filter_fs(X, Y)
        print(X)
    elif process == "wrapper":
        X = wrapper_fs(X, Y)
        print(X)



    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    my_score = []
    true_score = []
    for train_index, test_index in cv.split(X):
        X_train, X_test, Y_train, Y_test = X.iloc[train_index], X.iloc[test_index], Y[train_index], Y[test_index]

        Y_pred = nb(X_train, Y_train, X_test)
        my_score.append(f1_score(Y_test, Y_pred))
        true_score.append(test(X_train, Y_train, X_test, Y_test))

    print(np.mean(my_score))
    # print(np.mean(true_score))


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
    # print(np.mean(true_score))


def filter_fs(X, Y):
    model = SelectKBest(chi2, k=14)
    new = model.fit(X, Y)
    print(X.shape)
    X_new = new.transform(X)
    print(X_new.shape)
    return pd.DataFrame(X_new)


def wrapper_fs(X, Y):
    features = forward_selection(X, Y)
    print(X.shape)
    X = X.drop([feature for feature in X.columns if feature not in features], axis=1)
    print(X.shape)
    return X


def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features) > 0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features, dtype='float64')
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(
                data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value < significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features

demo_numerical()
demo_categorical()

# demo_numerical()
# demo_numerical(process="filter")
# demo_numerical(process="wrapper")
# demo_numerical(process="pca")
# demo_numerical(process="lda")