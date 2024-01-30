import sys

sys.path.append('../python')

import phd.scattering.config as config

config.set_precision('single')
config.ENABLE_DS = False

from phd.scattering.sep_ws import MorletSampler1D, optimise_T
import matplotlib.pyplot as plt
import numpy as np


fs = 1
T = optimise_T(64, fs)

Q = 1

fb = MorletSampler1D(Q, T, fs)

print(fb.d_lambda)

# plt.plot(fb.phi)
# plt.show()

# plt.plot(np.real(fb.psi[0, :]))
# plt.plot(np.imag(fb.psi[0, :]))
# plt.show()

f = np.arange(fb.psi.shape[1])/fb.psi.shape[1]*fs
plt.plot(f, np.abs(np.fft.fft(fb.psi, axis=1)).T)
f = np.arange(fb.phi.shape[1])/fb.phi.shape[1]*fb.fs_psi_out
plt.plot(f, np.abs(np.fft.fft(fb.phi, axis=1)).T, '--')
plt.show(block=True)