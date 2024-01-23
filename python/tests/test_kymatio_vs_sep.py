import numpy as np
from kymatio.torch import Scattering2D
import torch
from time import time

# Set the parameters of the scattering transform.
J = 3
M, N = 64, 64

# Generate a sample signal.
x = torch.from_numpy(np.random.randn(1024, M, N).astype(np.float32))

# Define a Scattering2D object.
S = Scattering2D(J, (M, N))

# Calculate the scattering transform.
t0 = time()
Sx = S.scattering(x)
t1 = time()

print(t1 - t0)

print(Sx.shape)



import sys

sys.path.append('../python')

import phd.scattering.config as config
# config.MORLET_DEFINITION = config.MORLET_DEF_DEFAULT
config.MORLET_DEFINITION = config.MorletDefinition(2, 3, 2, 2, 3)
config.set_precision('single')
config.ENABLE_DS = True

from phd.scattering.sep_ws import SeperableWaveletScattering, optimise_T
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

torch.cuda.empty_cache()


fs = [1, 1]
Q = [[1, 1], [1, 1]]
T = [optimise_T(32, 1, eps=0.05)]*2
print(T)

ws = SeperableWaveletScattering(Q, T, fs, [1, 2], True)

t0 = time()
Sx = ws.scatteringTransform(x)
t1 = time()

print(t1 - t0)

print(Sx.shape)