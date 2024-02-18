import sys

sys.path.append('../python')

import phd.scattering.config as config
config.MORLET_DEFINITION = config.MorletDefinition(2, 3, 2, 3, 4)
# config.MORLET_DEFINITION = config.MORLET_DEF_PERFORMANCE
config.set_precision('single')
config.ENABLE_DS = True

from phd.scattering.sep_ws import SeperableScatteringLayer, optimise_T
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

import skimage as ski

im = ski.data.binary_blobs().astype(config.NUMPY_REAL)/256
print(im.shape)
# im = ski.transform.downscale_local_mean(im, (2,2,1))
# if len(im.shape) > 2:
#     im = ski.color.rgb2gray(im).astype(config.NUMPY_REAL)

print(im.shape)
im_torch = torch.from_numpy(im)

Nx = im.shape[0:2]

torch.cuda.empty_cache()

fs = [1, 1]
Q = [1, 1]
T = [optimise_T(128, 1)]*2
print(T)

wsl = SeperableScatteringLayer(Q, T, fs, [0, 1], Nx, include_on_axis_wavelets=False)


# plt.subplot(Np, Np, 1)
# plt.imshow(x.cpu())

# for i in range(min(U.shape[-1], Np*Np-1)):
#     plt.subplot(Np, Np, i + 1)
#     plt.imshow(U.cpu()[:, :, i].real)
#     lambdas = wsl.filter_lambda_pairs[i]
#     plt.title("{:.2f}, {:.2f}".format(*lambdas))

# plt.show()


t0 = time()
# idx, _ = wsl.select_filters()
U, S = wsl.US(im_torch.cuda())
torch.cuda.synchronize()
t1 = time()
print(f"Took {t1 - t0} s")



# print(U.shape)
# print(S.shape)

Np = 7

plt.subplot(Np, Np, 1)
plt.imshow(im, cmap='gray')

for i in range(min(S.shape[-1], Np*Np-1)):
    plt.subplot(Np, Np, i + 2)
    im_filt = S.cpu()[..., i]
    im_filt /= torch.max(im_filt)
    plt.imshow(im_filt, cmap='gray')
    # lambdas = wsl.filter_lambda_pairs[i]
    # plt.title("{:.2f}, {:.2f}".format(*lambdas))

plt.show()





