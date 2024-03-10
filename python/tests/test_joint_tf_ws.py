import sys

sys.path.append('../python')

import phd.scattering.config as config
config.set_precision('single')

from phd.scattering.sep_ws import JointTFScattering, optimise_T
from phd.scattering.test_signals.fm import fm_sin
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time



fs = 14700
N = fs*5

Tt = optimise_T(0.25, fs)
Ttf = optimise_T(16, 1)
Qt = 4
Qtf = [[2,1], [1,1]]

t = np.arange(N)/fs
x = fm_sin(fs, N, 200, 1, 100)

jtfws = JointTFScattering(Qt, Qtf, Tt, Ttf, fs, 1, N, fstart=20)



s = jtfws.scatteringTransform(x)

print(s.shape)

y = s.numpy()[0, :, :]

plt.plot(y)
plt.show(block=True)
