import numpy as np

def _time_series_fun(X, f):
    mus = []
    for xn in X:
        mus.append(f(xn, axis=0))
    return np.mean(mus, axis=0)

def time_series_mean(X):
    return _time_series_fun(X, np.mean)

def time_series_std(X):
    return _time_series_fun(X, np.std)

def time_series_normalise(X, mu, std):
    Xnorm = []
    for xn in X:
        Xnorm.append((xn - mu)/std)
    return Xnorm   

def time_series_select_feat_variance_thresh(X, k):
    variances = time_series_std(X)**2
    idx = np.argsort(variances)[::-1] #sort descending
    return idx[:k]

def time_series_select_features(X, idx):
    X_new = []
    for x in X:
        X_new.append(x[:, idx])
    return X_new