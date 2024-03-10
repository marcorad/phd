import pickle as pkl

from phd.dataprocessing.esc50 import load_esc50

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from torch import Tensor

from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, svm

import numpy as np

from tqdm import tqdm
from typing import List, Dict

def split(X: List[np.ndarray], labels, test_size = 0.2):
    classes = {}
    N = len(X)
    for n in range(N):
        y = labels[n]
        if y not in classes.keys():
            classes[y] =[]
        x = X[n]
        classes[y].append(x)        
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for v in classes.values():
        np.random.shuffle(v)
    for l, d in classes.items():
        n_split = int(len(d)*(1-test_size))
        for i, x in enumerate(d):
            if i < n_split:
                X_train.append(x)
                y_train.append(l)
            else:
                X_test.append(x)
                y_test.append(l)
    return X_train, y_train, X_test, y_test  

def remove_silence(X: np.ndarray, EPS=0.05e-6):  
    N = X.shape[0]  
    X = X.reshape(*X.shape[0:2], -1)    
    X_list = []    
    for n in range(N):
        x = X[n, :, :]
        E = np.sqrt(np.mean(x * x, axis=1)) #calculate the RMS across the features
        keep_idx = E >= EPS
        x = x[keep_idx, :]
        X_list.append(x)   
    return X_list

def fit_model(X_train, y_train, clf):
    y = []
    for i in range(len(X_train)):
        y += [y_train[i]]*len(X_train[i])
    X_train = np.concatenate(X_train, axis=0)
    clf.fit(X_train, y)

def transform_model(X, model):
    X_t = []
    for xn in X:
        X_t.append(model.transform(xn))
    return X_t

def predict_model(X, clf: svm.SVC):
    y_pred = []
    classes = clf.classes_
    for xn in X:
        p = np.sum(np.log(clf.predict_proba(xn)), axis=0)
        y_pred.append(classes[np.argmax(p)])
    return y_pred

def time_series_fun(X, f):
    mus = []
    for xn in X:
        mus.append(f(xn, axis=0))
    return np.mean(mus, axis=0)

def time_series_mean(X):
    return time_series_fun(X, np.mean)

def time_series_std(X):
    return time_series_fun(X, np.std)

def time_series_normalise(X, mu, std):
    Xnorm = []
    for xn in X:
        Xnorm.append((xn - mu)/std)
    return Xnorm       
    

fname_feats = 'data/ws-esc50-[023510204081632652, 160]-20-4-[[2, 2], [1, 1]].pkl'

with open(fname_feats, 'rb') as file:
    x: Tensor = pkl.load(file)

x = x.numpy()
print(x.shape)

print(x.shape)
_, labels = load_esc50()
X = remove_silence(x)
# print([len(x) for x in X])
X_train, y_train, X_test, y_test = split(X, labels, 0.2)


print(len(X_train), len(y_train), len(X_test), len(y_test))

mu = time_series_mean(X_train)
std = time_series_std(X_train)
X_train = time_series_normalise(X_train, mu, std)
X_test = time_series_normalise(X_test, mu, std)

priors=[1/50 for _ in range(50)]

feat_red = LinearDiscriminantAnalysis(solver='svd', priors=priors)
clf = svm.SVC(probability=True, verbose=True)
# clf = LinearDiscriminantAnalysis(solver='eigen', priors=priors)


fit_model(X_train, y_train, feat_red)
X_train = transform_model(X_train, feat_red)
X_test = transform_model(X_test, feat_red)

fit_model(X_train, y_train, clf)

y_pred = predict_model(X_test, clf)

print(
    f"Classification report for classifier {feat_red}:\n"
    f"{metrics.classification_report(y_test, y_pred, digits=3)}\n"
)


