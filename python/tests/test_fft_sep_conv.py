import sys


import torch
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../python")
import phd.scattering.config as config
config.set_precision('single')

from phd.scattering.conv import SeperableConv
from phd.scattering.morlet import sample_gauss

torch.cuda.empty_cache()


fs = 250.0

d1 = 2048
d2 = 128
d3 = 128

x = torch.randn((d1, d2, d3), dtype=config.TORCH_REAL)
# x = torch.zeros((d1, d2, d3), dtype=config.TORCH_REAL)
# x[:, d2//2, d3//2] = 1.0
# x[:, d2//4, d3//4] = 1.0
# x[:, d2//8, d3//8] = 1.0


Nh = 19
n = np.arange(-Nh//2 + 1, Nh//2 + 1)
print(n)
g = sample_gauss(n, Nh/2/3)
h = torch.zeros((1, Nh), dtype=config.TORCH_COMPLEX)
h[:, :] = torch.tensor(g, dtype=config.TORCH_COMPLEX)/5
conv1 = SeperableConv([h, h], [d2, d3], ds=[4, 4], conv_dim=[1, 2])


xp = conv1.add_padding(x, pad_mode='constant')

t0 = time.time()
#perform seperable convolution
y = torch.real(conv1.convolve(xp))
torch.cuda.synchronize()
t1 = time.time()
print("TOOK ", t1 - t0)
print(y.shape)
y = conv1.remove_padding(y)
print(y.shape)

print(torch.sum(y).cpu().item())






# plt.subplot(211)
# plt.imshow(x[:,:,0].cpu())
# plt.title('x')
# plt.subplot(212)
# plt.imshow(y[:,:,0, 0].cpu())
# plt.title('y')
# plt.show(block = True)

x = x.cpu()
y = y.cpu()

# print("X")
# for i in range(d2):
#     for j in range(d3):
#         print("{:.2f} ".format(x[0, i, j]), end='')
#     print()

# print("Y")
# for i in range(y.shape[1]):
#     for j in range(y.shape[2]):
#         print("{:.2f} ".format(y[0, i, j, 0, 0]), end='')
#     print()

plt.subplot(211)
plt.imshow(x[0, :, :].cpu(), cmap='grey')
plt.title('x')
plt.subplot(212)
plt.imshow(y[0, :, :, 0, 0].cpu(), cmap='grey')
plt.title('y')
plt.show(block = True)