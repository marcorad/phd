import sys
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../python")

import phd.scattering.scattering_1d as s1d
from phd.scattering.conv import Conv1D

torch.cuda.empty_cache()

fs = 250.0

L = int(fs*61)
N = 10
F = 50

x = torch.zeros((N, 1, F, L), dtype=s1d.TORCH_DATA_TYPE)

x[:, 0, :, 500] = 1.0
x[:, 0, :, 2000] = 1.0
x[:, 0, F//2, 0] = 1.0

Nh = 255
h = torch.zeros((2, Nh), dtype=s1d.TORCH_DATA_TYPE)
h[:, [Nh//2-1, Nh//2, Nh//2+1]] = torch.tensor([-0.5, 1.0, -0.5], dtype=s1d.TORCH_DATA_TYPE)
conv1 = Conv1D(h, conv_dim=-1, ds=1)


Nh = 25
h = torch.zeros((2, Nh), dtype=s1d.TORCH_DATA_TYPE)
h[:, [Nh//2-1, Nh//2, Nh//2+1]] = torch.tensor([-0.5, 1.0, -0.5], dtype=s1d.TORCH_DATA_TYPE)
conv2 = Conv1D(h, conv_dim=-2, ds=1)

t0 = time.time()
#perform seperable convolution
y = conv1.conv_multiple(x)
print(x.shape, y.shape)
y = conv2.conv_multiple(y[:, [0], :, :])
print(x.shape, y.shape)
torch.cuda.synchronize()
t1 = time.time()
print("TOOK ", t1 - t0)

print(torch.cuda.memory_summary())


plt.subplot(211)
plt.plot(np.arange(x.shape[-1]), torch.as_tensor(x[0, 0, 0, :], dtype=torch.float64).cpu())
plt.plot(np.arange(y.shape[-1]), torch.as_tensor(y[0, 0, 0, :], dtype=torch.float64).cpu())
plt.title('Time conv')
plt.subplot(212)
plt.plot(np.arange(x.shape[-2]), torch.as_tensor(x[0, 0, :, 0], dtype=torch.float64).cpu())
plt.plot(np.arange(y.shape[-2]), torch.as_tensor(y[0, 0, :, 0], dtype=torch.float64).cpu())
plt.title('Freq conv')
plt.show(block = True)