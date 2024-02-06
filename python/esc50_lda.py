import pickle as pkl

from phd.dataprocessing.esc50 import load_esc50

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from torch import Tensor

from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, svm

import numpy as np

from tqdm import tqdm

def remove_silence(x: np.ndarray, labels, shuffle=True):
    if shuffle:
        shuf_i = np.arange(x.shape[0])
        np.random.shuffle(shuf_i)
        x = x[shuf_i, :, :]
        labels = np.array(labels)[shuf_i].tolist()
    eps = 1e-6
    E = np.sqrt(np.mean(x * x, axis=2))
    # print(np.min(E), np.percentile(E, [5, 25, 50, 75, 95]))
    # print(np.max(E))
    data = []
    y = []
    N = x.shape[0]
    n_win = []
    for i in range(N):
        e = E[i, :]
        keep_idx = e > eps
        xi = x[i, keep_idx, :]
        data.append(xi)
        y += [labels[i]]*xi.shape[0]
        n_win += [xi.shape[0]]
        print(xi.shape)
    data = np.concatenate(data, axis=0)
    return data, y, n_win

def get_split_idx(n_win, test_size):
    i = int(np.floor(len(n_win)*(1-test_size)))
    return int(np.sum(n_win[0:i])), i

fname_feats = 'data/ws-esc50-2048-30-8-2-1.pkl'
fname_feats = 'data/ws-esc50-[09404081632653061, 320]-50-4-[1, 1]-[1, 1].pkl'

with open(fname_feats, 'rb') as file:
    x: Tensor = pkl.load(file)

x = x.numpy()

print(x.shape)
_, labels = load_esc50()
X, y, n_win = remove_silence(x, labels)

# X = X / (np.max(X, axis=1)[:, None])

print(X.shape, len(y), n_win)

split_idx, n_win_idx = get_split_idx(n_win, 0.2)

X_train, X_test, y_train, y_test = X[:split_idx, :], X[split_idx:, :], y[0:split_idx], y[split_idx:]

mu = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mu)/std
X_test = (X_test - mu)/std

print("x_train shape ", X_train.shape)

priors=[1/50 for _ in range(50)]

feat_red = LinearDiscriminantAnalysis(solver='svd', priors=priors)
clf = svm.SVC(probability=True, verbose=True)

feat_red.fit(X_train, y_train)
X_train = feat_red.transform(X_train)
X_test = feat_red.transform(X_test)

clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
predicted_probs = clf.predict_log_proba(X_test)
classes = clf.classes_

print(
    f"Classification report for classifier {feat_red}:\n"
    f"{metrics.classification_report(y_test, predicted, digits=3)}\n"
)

from collections import Counter

#take the most common prediction of each tested file
n_win_test = n_win[n_win_idx:]
y_test_idx = np.cumsum(np.array(n_win_test)).tolist()
y_test_idx = [0] + y_test_idx[:-1]
y_pred = []
y_true = []

for n, i in zip(n_win_test, y_test_idx):
    file_pred = np.sum(predicted_probs[i:(i+n)], axis=0)

    y_pred += [classes[np.argmax(file_pred)]]
    y_true += [y_test[i]]
    
n_correct = 0    
for i in range(len(y_pred)):
    print(f'{y_true[i]}: {y_pred[i]}')
    n_correct += 1 if y_true[i] == y_pred[i] else 0
    
print(
    f"Classification report for classifier {feat_red}:\n"
    f"{metrics.classification_report(y_true, y_pred, digits=3)}\n"
    
)


