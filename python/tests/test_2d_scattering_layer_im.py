import sys

sys.path.append('../python')

import phd.scattering.config as config
config.MORLET_DEFINITION = config.MORLET_DEF_DEFAULT
# config.MORLET_DEFINITION = config.MORLET_DEF_PERFORMANCE
config.set_precision('single')
config.ENABLE_DS = True

from phd.scattering.sep_ws import SeperableScatteringLayer, optimise_T
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

torch.cuda.empty_cache()

fs = [1, 1]
Q = [1, 1]
T = [optimise_T(128, 1)]*2
print(T)



wsl = SeperableScatteringLayer(Q, T, fs, [0, 1], include_on_axis_wavelets=False)


# plt.subplot(Np, Np, 1)
# plt.imshow(x.cpu())

# for i in range(min(U.shape[-1], Np*Np-1)):
#     plt.subplot(Np, Np, i + 1)
#     plt.imshow(U.cpu()[:, :, i].real)
#     lambdas = wsl.filter_lambda_pairs[i]
#     plt.title("{:.2f}, {:.2f}".format(*lambdas))

# plt.show()

import skimage as ski

im = ski.data.retina().astype(config.NUMPY_REAL)/256
print(im.shape)
im = ski.transform.downscale_local_mean(im, (8,8,1))
# if len(im.shape) > 2:
#     im = ski.color.rgb2gray(im).astype(config.NUMPY_REAL)

print(im.shape)
im_torch = torch.from_numpy(im)
U, S = wsl.US(im_torch)

print(U.shape)
print(S.shape)

Np = 7

plt.subplot(Np, Np, 1)
plt.imshow(im, cmap='gray')

for i in range(min(U.shape[-1], Np*Np-1)):
    plt.subplot(Np, Np, i + 2)
    im_filt = U.cpu()[..., i].norm(2, dim=2)
    # im_filt /= torch.max(im_filt)
    plt.imshow(im_filt, cmap='gray')
    lambdas = wsl.filter_lambda_pairs[i]
    plt.title("{:.2f}, {:.2f}".format(*lambdas))

plt.show()





