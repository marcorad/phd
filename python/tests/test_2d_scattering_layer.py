import sys

sys.path.append('../python')

import phd.scattering.config as config
# config.MORLET_DEFINITION = config.MorletDefinition(2, 3, 2, 3, 4)
# config.MORLET_DEFINITION = config.MORLET_DEF_PERFORMANCE
config.set_precision('single')
config.ENABLE_DS = True

from phd.scattering.scattering_layer import SeperableScatteringLayer
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

torch.cuda.empty_cache()

fs = [1, 1]
Q = [2, 2]
T = [64]*2
print(T)
Nx = 32

x = np.zeros((Nx, Nx), dtype=config.NUMPY_REAL)
x[Nx//2, Nx//2] = 1.0

print(x.shape)
x = torch.from_numpy(x).cuda()

wsl = SeperableScatteringLayer(Q, T, fs, [0, 1], [Nx, Nx], include_on_axis_wavelets=True)

print(wsl.paths)
print(len(wsl.paths))

U, S = wsl.US(x)

print("--------u---------")
for u in U.values():
    print(u.shape)
    
print("--------s---------")
for s in S.values():
    print(s.shape)












