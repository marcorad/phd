import sys

sys.path.append('../python')

import phd.scattering.config as config
config.set_precision('single')
config.ENABLE_DS = True
config.ENABLE_FREQ_DS = False

from phd.scattering.sep_ws import SeperableWaveletScattering, optimise_T
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

import skimage as ski

im = ski.data.retina().astype(config.NUMPY_REAL)/256
print(im.shape)
# im = ski.transform.downscale_local_mean(im, (2,2,1))
# if len(im.shape) > 2:
#     im = ski.color.rgb2gray(im).astype(config.NUMPY_REAL)

print(im.shape)
im_torch = torch.from_numpy(im)

Nx = im.shape[0:2]

torch.cuda.empty_cache()








fs = [1, 1]
Q = [[1, 1]] * 2
T = [optimise_T(64, 1)]*2
print(T)

ws = SeperableWaveletScattering(Q, T, fs, [0, 1], Nx, include_on_axis_wavelets=True, prune=False)


for l in ws.sws_layers:
    print("SCAT LAYER")
    print(l.get_psi_output_fs())
    print(l.get_psi_output_N())

import pprint
pp = pprint.PrettyPrinter(depth=10)
pp.pprint(ws.paths)

pp.pprint(ws.path_lambdas)

print("Number of paths: ", len(ws.paths.keys()))
print(any([k == None for k in ws.paths.items()]))

print("expected features ", len(ws.flattened_lambdas))

# plt.subplot(Np, Np, 1)
# plt.imshow(x.cpu())

# for i in range(min(U.shape[-1], Np*Np-1)):
#     plt.subplot(Np, Np, i + 1)
#     plt.imshow(U.cpu()[:, :, i].real)
#     lambdas = wsl.filter_lambda_pairs[i]
#     plt.title("{:.2f}, {:.2f}".format(*lambdas))

# plt.show()


t0 = time()
U, S = ws.scatteringTransform(im_torch, discard_U=False, normalise=True)
torch.cuda.synchronize()
t1 = time()
print(f"Took {t1 - t0} s")

import cProfile
cProfile.run('ws.scatteringTransform(im_torch); torch.cuda.synchronize()')



# print(U)
print(S.shape)

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





