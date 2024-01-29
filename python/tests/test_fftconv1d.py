import sys


import torch
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../python")
import phd.scattering.config as config
config.set_precision('single')


import phd.scattering.scattering_1d as s1d
from phd.scattering.conv import Conv1D

torch.cuda.empty_cache()


fs = 250.0

d1 = 2048
d2 = 17
d3 = 64

x = torch.zeros((d1, d2, d3), dtype=config.TORCH_REAL)



x[:, d2//2, :] = 1.0

Nh = 7
h = torch.zeros((2, Nh), dtype=config.TORCH_COMPLEX)
h[:, [Nh//2-2, Nh//2-1, Nh//2, Nh//2+1, Nh//2+2]] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=config.TORCH_COMPLEX)
conv1 = Conv1D(h, d2, ds=1, conv_dim=1)


xp = conv1.add_padding(x)

t0 = time.time()
#perform seperable convolution
y = torch.real(conv1.convolve(xp))
torch.cuda.synchronize()
t1 = time.time()
print("TOOK ", t1 - t0)
print(y.shape)
y = conv1.remove_padding(y)
print(y.shape)






# plt.subplot(211)
# plt.imshow(x[:,:,0].cpu())
# plt.title('x')
# plt.subplot(212)
# plt.imshow(y[:,:,0, 0].cpu())
# plt.title('y')
# plt.show(block = True)

plt.subplot(211)
plt.plot(x[25,:,0].cpu())
plt.title('x')
plt.subplot(212)
plt.plot(y[25,:,0, 0].cpu())
plt.title('y')
plt.show(block = True)