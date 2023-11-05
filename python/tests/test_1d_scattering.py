import sys
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../python")

import scattering.scattering_1d as s1d

fs = 250.0

L = int(fs*60*60)
N = 1

x = torch.randn((N, 1, L), dtype=s1d.torch_data_type)


scat = s1d.Scattering1D([12, 4, 1], 3.0, fs, fstart=15.0)

t0 = time.time()
res = scat.transform(x)
t1 = time.time()

print(t1 - t0)

scat.feature_vector(res)

# print(res.keys())