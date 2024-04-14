import sys
sys.path.append('../python')

import numpy as np
from numpy.fft import fftshift

from sepws.scattering.separable_scattering import SeparableScattering
import torch
from sepws.scattering.config import cfg

cfg.cuda()
cfg.set_beta(1, 2.5)
cfg.set_alpha(1, 3.0)


import matplotlib.pyplot as plt
from math import ceil

N = [512, 512]
J = 5
d = [2**J, 2**J]
Q = [[1,1]]

B = 100

x = torch.randn((B, N[0], N[1])).type(torch.float32).to(cfg.DEVICE)

from time import time

scat = SeparableScattering(N, d, Q, allow_ds=True)

t0 = time()
s = scat.scattering(x)
t1 = time()
torch.cuda.synchronize()
dt_mine = t1-t0
print(f'{dt_mine=: .2f} sec')

n_paths_mine = s.shape[1]
print(s.shape)

del s

torch.cuda.empty_cache()


from kymatio.torch import Scattering2D

kyscat = Scattering2D(J, N, max_order=len(Q))
kyscat.cuda()
t0 = time()
s = kyscat.scattering(x)
t1 = time()
n_paths_kymatio = s.shape[1]
print(s.shape)
torch.cuda.synchronize()
dt_kymatio = t1-t0
print(f'{dt_kymatio=: .2f} sec')
print(f'{n_paths_mine=} {n_paths_kymatio=}')
print(f'Relative speedup: {dt_kymatio / dt_mine * n_paths_mine / n_paths_kymatio : .2f}x')


