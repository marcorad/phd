import numpy as np
import matplotlib.pyplot as plt
Nx = 32
X = np.zeros(Nx)
k_end = 8
tri = np.flip(np.linspace(0, 1, k_end))
X[0:k_end] = tri
X[Nx-k_end+1:] = np.flip(tri[1:])
plt.figure()
plt.plot(X)


x = np.fft.ifft(X)
plt.figure()
plt.plot(x.real)
plt.plot(x.imag)

import numpy as np
from numpy.fft import fft, ifft

def subsample_fourier(xf, sub):  
    xfs = xf.reshape(sub, -1).mean(axis=0)
    print(xf.reshape(sub, -1).shape)
    return xfs


d = 2
Xd = subsample_fourier(X, d)
plt.figure()
plt.plot(Xd)
xr = X.reshape(d, -1)
plt.plot(xr.T)


a = np.arange(16)
print(a.reshape(d, -1))

# plt.show()
