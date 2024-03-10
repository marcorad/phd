import numpy as np


import sys
sys.path.append('../python')
import phd.scattering.config as config
# config.MORLET_DEFINITION = config.MorletDefinition(2,4,2,2,2)
config.MORLET_DEFINITION = config.MorletDefinition(2,4,2,2,2)
config.set_precision('single')
config.ENABLE_DS = True
from phd.scattering.sep_ws import optimise_T, SeperableWaveletScattering

from phd.dataprocessing.medmnist3d import load_train_test, DATASETS

import torch

Q = [[0.9]*3]
T = [optimise_T(48,1)]*3
print(T)
fs = [1, 1, 1]
dims = [1, 2, 3]
N = [28, 28, 28]
DCT = True

ws = SeperableWaveletScattering(Q, T, fs, dims, N)

for d in DATASETS:
    print(d)
    X_train, y_train, X_test, y_test, X_val, y_val = load_train_test(d, False)
    
    def normalise(X):
        # p = np.std(X, axis=(1, 2, 3), keepdims=True)
        # p[p==0] = 1
        # X /= p.astype(config.NUMPY_REAL)
        return X
    
    X_train = X_train.astype(config.NUMPY_REAL)/256
    X_train = normalise(X_train)
    
    X_test = X_test.astype(config.NUMPY_REAL)/256
    X_test = normalise(X_test)
    
    X_val = X_val.astype(config.NUMPY_REAL)/256
    X_val = normalise(X_val)
    
    s_train = ws.scatteringTransform(torch.from_numpy(X_train), batch_dim=0, batch_size=2, dct=DCT)
    print(s_train.shape)
    s_test = ws.scatteringTransform(torch.from_numpy(X_test), batch_dim=0, batch_size=2, dct=DCT)
    s_val = ws.scatteringTransform(torch.from_numpy(X_val), batch_dim=0, batch_size=2, dct=DCT)

    fname = f'ws-{d}-mnist3d-{Q=}-{T=}-{DCT=}.pkl'

    import pickle as pkl
    with open('data/' + fname, 'wb') as file:
        pkl.dump((s_train, y_train, s_test, y_test, s_val, y_val), file)

