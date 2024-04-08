import numpy as np


import sys
sys.path.append('../python')
import sepws.scattering.config as config
config.MORLET_DEFINITION = config.MorletDefinition(2,4,2,1,2)
# config.MORLET_DEFINITION = config.MorletDefinition(2,3,2,1,2)
config.set_precision('single')
config.ENABLE_DS = True
from sepws.scattering.sep_ws import optimise_T, SeperableWaveletScattering

from sepws.dataprocessing.organmist3d import load_train_test

import torch

Q = [[0.5]*3]*1
T = [optimise_T(24,1)]*3
print(T)
fs = [1, 1, 1]
dims = [1, 2, 3]
N = [28, 28, 28]

X_train, y_train, X_test, y_test = load_train_test(True)

ws = SeperableWaveletScattering(Q, T, fs, dims, N)
s_train = ws.scatteringTransform(torch.from_numpy(X_train), batch_dim=0, batch_size=8)
s_test = ws.scatteringTransform(torch.from_numpy(X_test), batch_dim=0, batch_size=8)
print(s_train.shape, s_test.shape)

fname = f'ws-organmnist3d-{Q=}-{T=}.pkl'

import pickle as pkl
with open('data/' + fname, 'wb') as file:
    pkl.dump((s_train, y_train, s_test, y_test), file)

