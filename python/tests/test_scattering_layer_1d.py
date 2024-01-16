import sys

sys.path.append('../python')

import phd.scattering.config as config
config.MORLET_DEFINITION = config.MORLET_DEF_BALANCED
config.set_precision('single')

from phd.scattering.sep_ws import ScatteringLayer1D, optimise_T
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

torch.cuda.empty_cache()

fs = 250
Q = 16
T = optimise_T(2, fs)
print(T)

wsl = ScatteringLayer1D(Q, T, fs)

print(torch.cuda.memory_allocated() / 1024 / 1024)

wslf = ScatteringLayer1D(Q, T, fs, full= True)

print(torch.cuda.memory_allocated() / 1024 / 1024)

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

Nx = fs*60*60*1
f0 = 50
n = np.arange(Nx)
x = np.sin(f0/fs*n*np.pi*2, dtype=config.NUMPY_REAL)
x = x[None, None, None, :]
x = torch.from_numpy(x)

print(torch.cuda.memory_allocated() / 1024 / 1024)


torch.cuda.synchronize()


t0 = time()
u1, s1 = wsl.US(x)
torch.cuda.synchronize()
t1 = time()
print('1 took {:.2f} secs'.format(t1 - t0))
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1024 / 1024)

t0 = time()
u2, s2 = wsl.US(x, bw_w=2*np.pi*60)
torch.cuda.synchronize()
t1 = time()
print('1 took {:.2f} secs'.format(t1 - t0))
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1024 / 1024)

t0 = time()
u3, s3 = wslf.US(x)
t1 = time()
print('1 took {:.2f} secs'.format(t1 - t0))
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1024 / 1024)

t0 = time()
u4, s4 = wslf.US(x, bw_w=2*np.pi*80)
t1 = time()
print('1 took {:.2f} secs'.format(t1 - t0))
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1024 / 1024)
# plt.imshow(s1[0, :, 0, :].cpu())
# plt.show(block=True)

torch.cuda.empty_cache()


