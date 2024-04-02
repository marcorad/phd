from kymatio.scattering2d.filter_bank import morlet_2d, gabor_2d, filter_bank

import matplotlib.pyplot as plt
import numpy as np

M, N = 40, 40 #padded MNIST
fb = filter_bank(M, N, 3)

phi = fb['psi'][18]['levels'][0]
phi = fb['phi']['levels'][0]

Phi = np.fft.fftshift(np.abs(phi), axes=(0,1))

Phi = Phi/np.max(Phi)

# Phi = 20*np.log10(Phi)

plt.imshow(Phi, extent=(-0.5, 0.5, -0.5, 0.5))
plt.show(block=True)

