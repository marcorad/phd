import torch
from torch import Tensor

import sys
sys.path.append('../python')

import numpy as np

from sepws.scattering.morlet import sample_gauss
from sympy.ntheory import factorint


def optimise_invariance_scale(isc, eps = 0.1):
    
    if isc == 1: return 1
    
    d1 = int(np.floor(isc * (1 - eps)))
    d2 = int(np.floor(isc * (1 + eps)))
    kmax = -1
    d_hat = d1

    for d in range(d1, d2 + 1):
        M = list(factorint(d).values())
        k = np.sum(M)
        if k > kmax:
            kmax = k
            d_hat = d
            
    return d_hat

def reflpad1d(x, d):
    N = x.shape[0]
    c = d - ((N + d) % (d))
    Npad = N + d + c
    print(f'{Npad=}, {Npad-c=}')
    xpad = np.zeros(Npad)
    l = d // 2
    xpad[l:l+N] = x
    xpad[0:l] = x[l:0:-1]
    xpad[N:N+l] = x[N-1:N-l-1:-1]
    return xpad

def unpad1d(x, Norig, d):
    return x[1:-1]


fs = 1
beta = 2.5
isc = 8
d_hat = optimise_invariance_scale(isc, 0.2)
print(f'{d_hat=}')
N = 29

x = np.zeros(N)
x[N//2] = 1.0
# x[0] = 1.0

xpad = reflpad1d(x, d_hat)

Npad = xpad.shape[-1]

sigma = d_hat / fs * beta / np.pi # sigma_w = pi * beta/d_hat; downsampling limits frequency to pi/d_hat; bw_to_sigma_ratio = sigma_w / (pi / d_hat) = beta
t = (np.arange(Npad) - Npad/2)/fs
h = sample_gauss(t, sigma) / fs

H = np.abs(np.fft.fft(h))

hp = np.real(np.fft.ifft(H))

X = np.fft.fft(xpad)

Y = X * H

Yd = Y.reshape(d_hat, Y.shape[0]//d_hat).mean(0)

yd = np.real(np.fft.ifft(Yd))
y = unpad1d(yd, N, d_hat)

print(f'{y.shape=}')

import matplotlib.pyplot as plt

plt.figure()
plt.subplot(211)
plt.plot(np.abs(Y))
plt.subplot(212)
plt.plot(np.abs(Yd))

plt.figure()
plt.plot(y)

plt.figure()
plt.plot(np.arange(Npad)/Npad, H)
plt.plot([0.5/d_hat]*2, [0, 1])
plt.show(block = True)


