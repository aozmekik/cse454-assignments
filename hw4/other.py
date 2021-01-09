from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

# TODO. 
# implement categorical
# implement f1 scores
# implement filter feature selection
# implement wrapper feature selection
# pca tool
# lda tool

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

def test_2(X_train, Y_train, X_val, Y_val):
    from sklearn.naive_bayes import CategoricalNB
    # for col in X_train.columns:
    #     X_train[col] = X_train[col].astype('category').cat.codes
    #     X_val[col] = X_val[col].astype('category').cat.codes

    model = CategoricalNB()
    model.fit(X_train, Y_train)
    print(accuracy_score(Y_val, model.predict(X_val)))

def cat(X_train, Y_train, X_val, Y_val):
    # column dist. FIXME. cols to count
    df = X_train.groupby(Y_train)
    count = [df[c].value_counts() for c in X_train.columns]

    distinct_cols = [len(X_train[col].unique()) for col in X_train.columns]
    
    # class distribution
    PCi = X_train.groupby(Y_train).apply(lambda x: len(x)) / X_train.shape[0]

    Y_pred = []
    for i in range(X_val.shape[0]): # row iterate
        p = {}

        for Ci in np.unique(Y_train): # class iterate
            p[Ci] = PCi.iloc[Ci]
            V = count[i]
            for j, val in enumerate(X_val.iloc[i]): # column iterate
                # applying laplace smooth 1.
                V = count[j][Ci].sum() + distinct_cols[j]
                p[Ci] *= ((count[j][Ci, val] + 1)  if (Ci, val) in count[j] else 1) / V

        Y_pred.append(pd.Series(p).values.argmax())
    
    print(accuracy_score(Y_val, Y_pred))




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
    

# data = load_iris()


# X, Y, column_names = data['data'], data['target'], data['feature_names']
# X = pd.DataFrame(X, columns=column_names)

# print(X.head())
# print(Y)


# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=44)

# con(X_train, Y_train, X_val, Y_val)

# test(X_train, Y_train, X_val, Y_val)

df = pd.read_csv('data/tennis.csv')

# df['play'] = df['play'].astype('category').cat.codes
for col in df.columns:
    df[col] = df[col].astype('category').cat.codes
print(df.head())

X = df.drop('play', axis=1)
Y = df['play']
Y = np.array(Y)


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=7)
print(X_train.head())
cat(X_train, Y_train, X_val, Y_val)
test_2(X_train, Y_train, X_val, Y_val)


# df = pd.read_csv('data/tennis.csv')

# # df['play'] = df['play'].astype('category').cat.codes
# for col in df.columns:
#     df[col] = df[col].astype('category').cat.codes

# X = df.drop('play', axis=1)
# Y = df['play']
# Y = np.array(Y)


# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=20)
# print(X_train.head())


# test_2(X_train, Y_train, X_val, Y_val)
