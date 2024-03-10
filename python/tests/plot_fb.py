import sys

sys.path.append('../python')

from phd.scattering import config
config.MORLET_DEFINITION = config.MorletDefinition(2,3,2,2,2)
config.set_precision('single')
from phd.scattering.sep_ws import optimise_T

from phd.scattering.morlet_sampler import MorletSampler1D

Q = 3
T = 16
fs = 1
print(optimise_T(T, fs))


sampler = MorletSampler1D(Q, T, fs, allow_seperate_ds=False)
sampler_n = MorletSampler1D(Q, T, fs, allow_seperate_ds=False, pol=-1)
psi = sampler.get_psi_filter_matrix()
psi_n = sampler_n.get_psi_filter_matrix()

import matplotlib.pyplot as plt
import numpy as np

Psi = np.abs(np.fft.fft(psi, axis=0))
Psi_n = np.abs(np.fft.fft(psi_n, axis=0))
N = psi.shape[0]
f = np.arange(N)/N * fs

plt.plot(f, Psi)
plt.plot(f, Psi_n, '--')
plt.xlabel('Frequency (Hz)')
plt.title(f'{Q=}, {T=}')
plt.show(block=True)
