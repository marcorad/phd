import sys

sys.path.append('../python')

import phd.scattering.config as config
# config.MORLET_DEFINITION = config.MORLET_DEF_DEFAULT
# config.MORLET_DEFINITION = config.MORLET_DEF_PERFORMANCE
config.set_precision('single')
config.ENABLE_DS = True

from phd.timeseries.sws_transform import time_series_sws_tranform
from phd.scattering.sep_ws import optimise_T, SeperableWaveletScattering
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time
from phd.dataprocessing import ucf


torch.cuda.empty_cache()

fs = [1, 1, 10]
Q = [[2,2,2]]
T = [optimise_T(48, fs[0]), optimise_T(48, fs[1]), optimise_T(0.8, fs[2])]
N = [67, 80, 128] #process 6.4 seconds at a time
kernel_size = 1
MP = torch.nn.MaxPool2d(kernel_size=kernel_size)




fname = f'ws-ucf-{Q=}-{T=}-{kernel_size=}.pkl'


# im = ski.transform.downscale_local_mean(im, (4,4))
# if len(im.shape) > 2:
#     im = ski.color.rgb2gray(im).astype(config.NUMPY_REAL)
# print(im.shape)

ws = SeperableWaveletScattering(Q, T, fs, [0, 1, 2], N, include_on_axis_wavelets=True)
X_train, y_train, X_test, y_test = ucf.read_train_test()

sl = slice(0, None)

x_train = X_train[sl]
y_train = y_train[sl]
x_test = X_test[sl]
y_test = y_test[sl]

print(X_train[0].shape)


def max_pool_and_select_regions(s: torch.Tensor):
    global MP
    #s = (Nx, Ny, Nt, Nfeat)
    s = s.swapaxes(0, 2) #(Nt, Ny, Nx, Nfeat)
    s = s.swapaxes(1, 3) #(Nt, Nfeat, Nx, Ny)
    sp: torch.Tensor =  MP(s)  #(Nt, Nfeat, Nx', Ny')
    #now only focus on 25% of the video which contains the most activity
    # n = int(0.25*sp.shape[2]*sp.shape[3]) 
    # sa = torch.mean(sp, dim=(0, 1)).reshape(-1) #(Nx', Ny') -> (Nxy')
    # ia = torch.argsort(sa, descending=True)
    # sr = sp.reshape(sp.shape[0], sp.shape[1], -1) #(Nt, Nfeat, Nxy')
    # sr = torch.index_select(sr, dim=2, index=ia[:n]).reshape(sr.shape[0], -1) #(Nt, Nfeat')
    return sp
    

S_train = time_series_sws_tranform(x_train, ws, time_dim=2, func=max_pool_and_select_regions, flatten=False)
S_test= time_series_sws_tranform(x_test, ws, time_dim=2, func=max_pool_and_select_regions, flatten=False)

print(S_train[0].shape)

import pickle as pkl
with open(f'data/{fname}', 'wb') as file:
    pkl.dump((S_train, y_train, S_test, y_test), file)








