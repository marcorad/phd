import sys

sys.path.append('../python')

import phd.scattering.config as config
# config.MORLET_DEFINITION = config.MorletDefinition(2, 3, 2, 3, 4)
# config.MORLET_DEFINITION = config.MORLET_DEF_PERFORMANCE
config.set_precision('single')
config.ENABLE_DS = True

from phd.scattering.sep_ws import SeperableScatteringLayer, optimise_T
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

torch.cuda.empty_cache()

fs = [1, 1]
Q = [1, 1]
T = [optimise_T(32, 1), optimise_T(32, 1)]
print(T)
Nx = 32

x = np.zeros((Nx, Nx), dtype=config.NUMPY_REAL)
x[Nx//2, Nx//2] = 1.0

print(x.shape)
x = torch.from_numpy(x)

wsl = SeperableScatteringLayer(Q, T, fs, [0, 1], [Nx, Nx], include_on_axis_wavelets=True)


idx, _ = wsl.select_filters()



U, S = wsl.US(x, idx)
print(U.shape, S.shape)

print(U.shape, S.shape)
Np = 7

plt.subplot(Np, Np, 1)
plt.imshow(x.cpu())

for i in range(min(U.shape[-1], Np*Np-1)):
    plt.subplot(Np, Np, i + 2)
    plt.imshow(U.cpu()[:, :, i].real)

plt.show()

for i in range(min(U.shape[-1], Np*Np-1)):
    plt.subplot(Np, Np, i + 2)
    plt.imshow(S.cpu()[:, :, i].real)

plt.show()







