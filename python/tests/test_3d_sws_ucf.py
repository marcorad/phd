import sys

sys.path.append('../python')

import phd.scattering.config as config
# config.MORLET_DEFINITION = config.MORLET_DEF_DEFAULT
# config.MORLET_DEFINITION = config.MORLET_DEF_PERFORMANCE
config.set_precision('single')
config.ENABLE_DS = True

from phd.timeseries.sws_transform import time_series_sws_tranform
from phd.scattering.sep_ws import optimise_T, SeperableWaveletScattering
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time
from phd.dataprocessing import ucf

torch.cuda.empty_cache()

fs = [1, 1, 10]
Q = [[1, 1, 1], [1, 1, 1]]
T = [optimise_T(32, 1), optimise_T(32, 1), optimise_T(1, 10)]
N = [65, 80, 64]


# im = ski.transform.downscale_local_mean(im, (4,4))
# if len(im.shape) > 2:
#     im = ski.color.rgb2gray(im).astype(config.NUMPY_REAL)
# print(im.shape)

ws = SeperableWaveletScattering(Q, T, fs, [0, 1, 2], N, include_on_axis_wavelets=True)
X, _, _, _ = ucf.read_train_test()
x = X[0:10]
S = time_series_sws_tranform(x, ws, time_dim=2)


print([s.shape for s in S])








