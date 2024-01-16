import sys

sys.path.append('../python')

import phd.scattering.config as config
# config.MORLET_DEFINITION = config.MORLET_DEF_BALANCED
config.set_precision('single')
config.MORLET_DEFINITION = config.MORLET_DEF_BALANCED

from phd.scattering.sep_ws import SeperableScatteringLayerND, optimise_T
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

torch.cuda.empty_cache()

fs = [1, 1]
Q = [1, 1]
T = [optimise_T(32, 1), optimise_T(32, 1)]
print(T)

# wsl = ScatteringLayer1D(Q, T, fs)


# f = np.arange(wsl.sampler.psi.shape[1])/wsl.sampler.psi.shape[1]*fs
# plt.plot(f, np.abs(np.fft.fft(wsl.sampler.psi, axis=1)).T)
# f = np.arange(wsl.sampler.phi.shape[1])/wsl.sampler.phi.shape[1]*wsl.sampler.fs_psi_out

# plt.plot(f, np.abs(np.fft.fft(wsl.sampler.phi, axis=1)).T, '--')
# plt.show()

# f = np.arange(wslf.sampler.psi.shape[1])/wslf.sampler.psi.shape[1]*fs
# plt.plot(f, np.abs(np.fft.fft(wslf.sampler.psi, axis=1)).T)
# f = np.arange(wslf.sampler.phi.shape[1])/wslf.sampler.phi.shape[1]*wslf.sampler.fs_psi_out
# plt.plot(f, np.abs(np.fft.fft(wslf.sampler.phi, axis=1)).T, '--')
# plt.show(block=True)

#TODO VERTICAL IS MISSING! FIX THIS!

Nx = 32
f0 = 1/32
nx = np.arange(Nx)[:, None]
ny = (np.arange(Nx)[:, None]).T
print(nx.shape, ny.shape)
x = np.sign(np.sin(f0*nx*np.pi*2, dtype=config.NUMPY_REAL) * np.sin(f0*ny*np.pi*2*2, dtype=config.NUMPY_REAL))
x = x[None, None, :, :]

print(x.shape)
x = torch.from_numpy(x)

wsl = SeperableScatteringLayerND(Q, T, fs, [2, 3])

print(wsl.conv_layers[0].sampler.d_lambda)

U, S = wsl.US(x)
print(U.shape, S.shape)

plt.subplot(5, 5, 1)
plt.imshow(x[0, 0, :, :])

for i in range(min(U.shape[1], 24)):
    plt.subplot(5, 5, i + 2)
    plt.imshow(U[0, i, :, :].cpu())

plt.show()





