import sys
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../python")

import phd.scattering.scattering_1d as s1d

fs = 250.0

L = int(fs*60*60)
N = 10

x = torch.zeros((N, 1, L), dtype=s1d.TORCH_DATA_TYPE)

x[0, 0, 0] = 1.0


scat = s1d.Scattering1D([16, 4, 2], 2.0, fs, fstart=15.0, oversample=1)




t0 = time.time()
res = scat.transform(x)
torch.cuda.synchronize()
t1 = time.time()
dt = t1 - t0
print("TOOK ", dt, f" seconds {dt/L/N*fs*60*60} per hour of audio")

for p, r in res.items(): print(p, r[0].shape, r[1].shape)

for fb in scat.filter_banks:
    print(fb.config.lpf_downsampling_factor, fb.config.filter_downsampling_factor)
print("Time taken ", t1 - t0)
print("TRUE T ", scat.T)

print(scat.module_indices)


scat.feature_vector(res)

# x1 = np.abs(res[(0,)][1][:, -1, :].numpy())[0, :]
# x2 = np.abs(res[(0,0)][1][:, 1, :].numpy())[0, :]

# plt.subplot(211)
# plt.plot(x1)
# plt.subplot(212)
# plt.plot(x2)
# plt.show(block = True)

# scat.filter_banks[0].plot()
# scat.filter_banks[1].plot()

    