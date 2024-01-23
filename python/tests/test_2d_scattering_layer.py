import sys

sys.path.append('../python')

import phd.scattering.config as config
config.MORLET_DEFINITION = config.MORLET_DEF_BALANCED
# config.MORLET_DEFINITION = config.MORLET_DEF_PERFORMANCE
config.set_precision('double')
config.ENABLE_DS = False

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

# wsl = ScatteringLayer1D(Q, T, fs)


# f = np.arange(wsl.sampler.psi.shape[1])/wsl.sampler.psi.shape[1]*fs
# plt.plot(f, np.abs(np.fft.fft(wsl.sampler.psi, axis=1)).T)
# f = np.arange(wsl.sampler.phi.shape[1])/wsl.sampler.phi.shape[1]*wsl.sampler.fs_psi_out

# plt.plot(f, np.abs(np.fft.fft(wsl.sampler.phi, axis=1)).T, '--')
# plt.show()

# f = np.arange(wslf.sampler.psi.shape[1])/wslf.sampler.psi.shape[1]*fs
# plt.plot(f, np.abs(np.fft.fft(wslf.sampler.psi, axis=1)).T)
# f = np.arange(wslf.sampler.phi.shape1])/wslf.sampler.phi.shape[1]*wslf.sampler.fs_psi_out
# plt.plot(f, np.abs(np.fft.fft(wslf.sampler.phi, axis=1)).T, '--')
# plt.show(block=True)

#TODO VERTICAL IS MISSING! FIX THIS!

Nx = 31
f0 = 1/32
nx = np.arange(Nx)[:, None]
ny = (np.arange(Nx)[:, None]).T
print(nx.shape, ny.shape)
x = np.sign(np.sin(f0*nx*np.pi*2, dtype=config.NUMPY_REAL) * np.sin(f0*ny*np.pi*2, dtype=config.NUMPY_REAL))

x = np.zeros((Nx, Nx), dtype=config.NUMPY_REAL)
x[Nx//2, Nx//2] = 1.0

print(x.shape)
x = torch.from_numpy(x)

wsl = SeperableScatteringLayer(Q, T, fs, [0, 1], include_on_axis_wavelets=False)

U, S = wsl.US(x, nonlin=None)
print(U.shape, S.shape)

Np = 7

plt.subplot(Np, Np, 1)
plt.imshow(x.cpu())

for i in range(min(U.shape[-1], Np*Np-1)):
    plt.subplot(Np, Np, i + 1)
    plt.imshow(U.cpu()[:, :, i].real)
    lambdas = wsl.filter_lambda_pairs[i]
    plt.title("{:.2f}, {:.2f}".format(*lambdas))

plt.show()







