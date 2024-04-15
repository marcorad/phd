import sys
sys.path.append('../python')

import numpy as np
from numpy.fft import fftshift

from sepws.scattering.filterbank import scattering_filterbanks, calculate_padding_1d
from sepws.scattering.config import cfg

cfg.set_beta(1.0, 2.5)
cfg.set_alpha(1.0, 2.5)


N = 28 #MNIST size
d = 8
l, r, Nt = calculate_padding_1d(N, d)
print(l, r, Nt, Nt % d, Nt // d - 2)


oversample = 10
fb = scattering_filterbanks([Nt * oversample]*2, [d]*2, [[1]*2]*2,startfreq=None, allow_ds=False)

import matplotlib.pyplot as plt

d_in = 1
level = 1

# f = (np.arange(0, Nt*oversample//ds) - Nt * oversample / 2 / ds) / Nt / oversample

phi = fb[level][0][d_in]['phi'][:, None] * fb[level][1][d_in]['phi'][None, :]
phi = fftshift(phi)

L = [0.71**2]
extent = (-0.5, 0.5, -0.5, 0.5)
plt.contour(phi, levels = L, linestyles='dashed', extent=extent)

for l1 in fb[level][0][d_in]['psi'].keys():
    for l2 in fb[level][1][d_in]['psi'].keys():
        psi = fb[level][0][d_in]['psi'][l1][:, None] * fb[level][1][d_in]['psi'][l2][None, :]
        psi = fftshift(psi)
        plt.contour(psi, levels = L, extent=extent)
     
for l1 in fb[level][0][d_in]['psi'].keys():
    psi = fb[level][0][d_in]['psi'][l1][:, None] * fb[level][1][d_in]['phi'][None, :]
    psi = fftshift(psi)
    plt.contour(psi, levels = L, extent=extent)
    
for l2 in fb[level][1][d_in]['psi'].keys():
    psi = fb[level][0][d_in]['phi'][:, None] * fb[level][1][d_in]['psi'][l2][None, :]
    psi = fftshift(psi)
    plt.contour(psi, levels = L, extent=extent)
        
    

plt.show(block=True)    