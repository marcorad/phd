import sys

sys.path.append('../python')

import phd.scattering.config as config
# config.MORLET_DEFINITION = config.MORLET_DEF_DEFAULT
# config.MORLET_DEFINITION = config.MORLET_DEF_PERFORMANCE
config.set_precision('single')
config.ENABLE_DS = True

from phd.scattering.sep_ws import SeperableWaveletScattering, optimise_T
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

torch.cuda.empty_cache()

fs = [1, 1, 10]
Q = [[1, 1, 1], [0.5, 0.5, 0.5]]
T = [optimise_T(32, 1), optimise_T(32, 1), optimise_T(2, 10)]
N = [65, 80, 40]
B = 2 #batch size

vid = np.random.randn(B, *N).astype(np.float32)
# im = ski.transform.downscale_local_mean(im, (4,4))
# if len(im.shape) > 2:
#     im = ski.color.rgb2gray(im).astype(config.NUMPY_REAL)
# print(im.shape)

ws = SeperableWaveletScattering(Q, T, fs, [1, 2, 3], N, include_on_axis_wavelets=True)
vid_torch = torch.from_numpy(vid)
S = ws.scatteringTransform(vid_torch, discard_U=True)


print(S.shape)








