import sys

sys.path.append('../python')

import phd.scattering.config as config
# config.MORLET_DEFINITION = config.MORLET_DEF_DEFAULT
# config.MORLET_DEFINITION = config.MORLET_DEF_PERFORMANCE
config.MORLET_DEFINITION = config.MorletDefinition(2, 2, 2, 3, 3)
config.set_precision('single')
config.ENABLE_DS = True

from phd.scattering.sep_ws import SeperableWaveletScattering, optimise_T
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

torch.cuda.empty_cache()

fs = [1, 1]
Q = [[1, 1]]*3
T = [optimise_T(32, 1)]*2
print(T)

Nx = 32

ws = SeperableWaveletScattering(Q, T, fs, [0,1], [Nx, Nx], include_on_axis_wavelets=False)

import pprint
pp = pprint.PrettyPrinter(depth=10)
pp.pprint(ws.paths)

print("Number of paths: ", len(ws.paths.keys()))
print(any([k == None for k in ws.paths.items()]))

for l in ws.sws_layers:
    print(l.get_psi_output_fs())
    print(l.get_psi_output_N())


# plt.subplot(Np, Np, 1)
# plt.imshow(x.cpu())

# for i in range(min(U.shape[-1], Np*Np-1)):
#     plt.subplot(Np, Np, i + 1)
#     plt.imshow(U.cpu()[:, :, i].real)
#     lambdas = wsl.filter_lambda_pairs[i]
#     plt.title("{:.2f}, {:.2f}".format(*lambdas))

# plt.show()






