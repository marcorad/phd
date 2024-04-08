import sys
sys.path.append('../python')

import numpy as np
from numpy.fft import fftshift

from sepws.scattering.filterbank import _filterbank_1d, calculate_padding_1d


N = 28 #MNIST size
d = 8
l, r, Nt = calculate_padding_1d(N, d)
print(l, r, Nt, Nt % d, Nt // d - 2)


oversample = 10
fb, ds = _filterbank_1d(Nt * oversample, d, 2,startfreq=None, include_negative_lambdas=True, input_ds_factors=[1, 2, 4])

import matplotlib.pyplot as plt

ds = 1

f = (np.arange(0, Nt*oversample//ds) - Nt * oversample / 2 / ds) / Nt / oversample

plt.plot(f, fftshift(fb[ds]['phi']), '--')

for psi in fb[ds]['psi'].values():
    plt.plot(f, fftshift(psi))

plt.show(block=True)    