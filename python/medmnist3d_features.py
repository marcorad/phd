import numpy as np


import sys
sys.path.append('../python')
from sepws.scattering.config import cfg
cfg.cuda()

from sepws.scattering.separable_scattering import SeparableScattering

from sepws.dataprocessing.medmnist3d import load_train_test, DATASETS
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import affine_transform

import torch

def augment_train(X, y, tot_examples_in_each_class, keep_prop = True):
    y_unique = set(y)
    X_classes = {}
    counts = {}
    for yu in y_unique:
        x = X[y == yu, ...]
        X_classes[yu] = x
        counts[yu] = x.shape[0]
    max_counts = np.max(list(counts.values()))
    assert(max_counts < tot_examples_in_each_class)
    
    n_target = {}
    for yu in y_unique:
        if keep_prop:
            n_target[yu] = int(tot_examples_in_each_class * counts[yu] / max_counts)
        else:
            n_target[yu] = tot_examples_in_each_class
    
    for yu in y_unique:
        x = X_classes[yu]
        n = x.shape[0]
        x_aug = []
        i = 0
        while n + i < n_target[yu]:
            rot_lim = 45
            rmat = R.from_euler('xyz', (np.random.rand(3))*rot_lim*2 - rot_lim, degrees=True).as_matrix() #random rotation of +-rot_lim  
            o = np.array([14, 14, 14])
            o = o - o.dot(rmat.T)    
            x_aug.append(affine_transform(x[i % n, ...], rmat, offset=o, mode='reflect', order=5)[None, ...]) #rotate about center
            i += 1
        X_classes[yu] = np.concatenate([X_classes[yu]]+x_aug, axis=0)
    X_aug = []
    y_aug = []
    for yu in y_unique:
        X_aug.append(X_classes[yu])
        y_aug.append(np.array([yu]*n_target[yu]))
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)
        
    

Q = [[2]*3]
d = [4]*3
N = [28, 28, 28]
AUG = False

ws = SeparableScattering(N, d, Q)



for d in DATASETS:
    print(d)
    X_train, y_train, X_test, y_test, X_val, y_val = load_train_test(d, False)
    
    def normalise(X):
        # p = np.std(X, axis=(1, 2, 3), keepdims=True)
        # p[p==0] = 1
        # X /= p.astype(config.NUMPY_REAL)
        return X
    
    X_train = X_train.astype(np.float32)/256
    X_train = normalise(X_train)
    if AUG: X_train, y_train = augment_train(X_train, y_train, 200 if d == 'organ' else 2000)
    
    X_test = X_test.astype(np.float32)/256
    X_test = normalise(X_test)
    
    X_val = X_val.astype(np.float32)/256
    X_val = normalise(X_val)
    torch.cuda.empty_cache()
    norm = False
    s_train = ws.scattering(torch.from_numpy(X_train).type(cfg.REAL_DTYPE).cuda(), norm)
    print(s_train.shape)
    s_test = ws.scattering(torch.from_numpy(X_test).type(cfg.REAL_DTYPE).cuda(), norm)
    s_val = ws.scattering(torch.from_numpy(X_val).type(cfg.REAL_DTYPE).cuda(), norm)

    fname = f'ws-{d}-mnist3d-{Q=}.pkl'

    import pickle as pkl
    with open('data/' + fname, 'wb') as file:
        pkl.dump((s_train, y_train, s_test, y_test, s_val, y_val), file)

