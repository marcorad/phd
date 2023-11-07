import sys
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../python")

import scattering.scattering_1d as s1d
from scattering.test_signals.fm import fm_sin, fm_tri

fs = 250.0

config = s1d.ScatteringFB1DConfig(16, 8.0, fs, fstart=15, approximation_support=3.0, downsample_mode=s1d.DownsampleMode.OPTIMAL_T, oversampling_factor=1)

fb = s1d.ScatteringFB1D(config)

print("Output FS ", fb.config.lpf_output_fs, "Effective T ", fb.config.T)



conv = s1d.ScatteringFB1DModule(fb)

# L = int(1.4079*250*1)
L = 10000
n = torch.arange(L)
t = n / fs
dw = torch.cumsum(2*np.pi*50 + 2*np.pi*(20.0*torch.sin(2*np.pi*t*0.3)), 0) / fs
N = 1

x = torch.zeros((N, 1, L), dtype=s1d.torch_data_type)

x[0, 0, L//2] = 1.0
x[0, 0, :] = torch.sin(dw)

x = fm_tri(fs, L, 50, 0.3, 20, N) + fm_sin(fs, L, 100, 0.5, 10, N)





t0 = time.time()
u, s = conv.US(x)
t1 = time.time()
print(t1 - t0)

u = u.detach().numpy()[0, :, :]
s = s.detach().numpy()[0, :, :]

# fb.plot()

plt.subplot(121)
plt.imshow(u, extent=(0, 1, 0, 1))
plt.subplot(122)
plt.imshow(s, extent=(0, 1, 0, 1))
plt.show(block=True)
# fb.plot()