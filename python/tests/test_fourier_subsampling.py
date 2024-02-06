import sys
sys.path.append('../python')

from phd.scattering.morlet import sample_gauss
import torch
import matplotlib.pyplot as plt
import numpy as np

fs = 1000
sigma_t = 0.002
N = 500
t = torch.arange(-N, N+1).cpu().numpy()/fs
x = sample_gauss(t, sigma_t)*np.exp(t*1j*2*np.pi*350)

X = torch.fft.fft(torch.from_numpy(x))


X = torch.abs(X)
ds = 4

shape = X.shape

Xr = X.unfold(dimension=-1, size=shape[0]//ds, step=ds)

print(Xr.shape)
Xr = Xr.mean(-1, keepdim=False)
print(Xr.shape)

plt.plot(Xr)
plt.show()
