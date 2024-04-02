import sys
sys.path.append('../python')

from phd.scattering.morlet import sample_gauss
import torch
import matplotlib.pyplot as plt
import numpy as np

fs = 1000
sigma_t = 0.008
N = 500
t = torch.arange(-N, N).cpu().numpy()/fs
x = sample_gauss(t, sigma_t)*np.exp(t*1j*2*np.pi*250)

X = torch.fft.fft(torch.from_numpy(x))


# X = torch.abs(X)
ds = 4

shape = X.shape

Xr = X.view(ds, X.shape[0]//ds)



Xr = Xr.mean(0, keepdim=False)

print(Xr.shape)

xr = torch.fft.ifft(Xr)

plt.subplot(311)
plt.plot(np.real(x))
plt.plot(np.imag(x))
plt.plot(np.abs(x))
plt.subplot(312)
plt.plot(xr.real)
plt.plot(xr.imag)
plt.plot(xr.abs())
plt.subplot(313)
plt.plot(X.abs())
plt.plot(Xr.abs())
plt.show()
