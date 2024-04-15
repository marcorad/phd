import sys
sys.path.append('../python')

import numpy as np
from numpy.fft import fftshift

from sepws.scattering.separable_scattering import SeparableScattering
import torch
from sepws.scattering.config import cfg

cfg.cpu()
cfg.set_beta(1.0, 2.0)
cfg.set_alpha(1.0, 2.5)


import matplotlib.pyplot as plt
from math import ceil

import skimage.data as testim 
from skimage.color import rgb2gray

I = rgb2gray(testim.coffee())

N = list(I.shape)
d = [16,16]
Q = [[1,1]]

x = torch.zeros((1, N[0], N[1])).type(torch.float32).to(cfg.DEVICE)
x = torch.from_numpy(I)[None,...].type(torch.float32).to(cfg.DEVICE)



# x[0, 14, 14] = 1.0
# for i in range(28):
#     x[0, i, i] = 1.0
    
# for i in range(28):
#     x[0, 27-i, i] = 1.0

# x[0, 10:18, 10:18] = 1.0
    
plt.figure()
plt.imshow(x[0,...])



scat = SeparableScattering(N, d, Q, allow_ds=True)

import pprint

pprint.pprint(scat._calculate_paths())

s, sp, up, xpad = scat._scattering(x, True, True)

plt.figure()
plt.imshow(xpad[0,...])

u = list(up.values())

p = list(up.keys())

sum = 0

for si in s:
    sum += (si[0, 4:33, 4:33]**2).sum() * 28
    
print(len(s), sum, (x**2).sum())



plt.figure()

pl = list(sp.values())
titl = list(sp.keys())

titlr = []

for t in titl:
    print(t)
    if type(t[0]) != tuple:
        titlr.append([round(l, 2) for l in t])
    else:
        titlr.append([[round(l, 2) for l in k] for k in t])

Ns = min(len(pl), 300)
K = ceil(Ns**0.5)

for i in range(Ns):
    plt.subplot(K, K, i+1)
    # plt.imshow(pl[i][0, 4:33, 4:33])#, vmin=0, vmax=torch.concat(s).max())
    plt.imshow(pl[i][0, ...])#, vmin=0, vmax=torch.concat(s).max())
    plt.title(titlr[i])
    
plt.show(block=True)