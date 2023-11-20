import sys
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../python")

import phd.scattering.scattering_tf as stf
import phd.scattering.test_signals.fm as fm
from pprint import pprint
from time import time
from cycler import cycler

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0
plt.rcParams["axes.zmargin"] = 0
plt.rcParams["axes.zmargin"] = 0
plt.rcParams["axes.grid"] = False
plt.rcParams["image.interpolation"] = 'nearest'
plt.rcParams["image.cmap"] = 'magma'
plt.rcParams["image.aspect"] = 1
plt.rcParams["axes.prop_cycle"] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])







# device = torch.device('cuda: 0' if torch.cuda.is_available() else "cpu")

fs = 250
L = fs*60
N = 1

x = fm.fm_sin(fs, L, 30, 0.1, 5, N)
x += fm.fm_tri(fs, L, 80, 0.08, 10, N)
x.to(stf.DEVICE)

scat = stf.ScatteringTF(16, 4, 0.25, fs, Q_tf=[(1, 1)], oversample=1, fstart=15)
print(torch.cuda.memory_summary())

t0 = time()
res = scat.transform(x)
torch.cuda.synchronize()
t1 = time()
dt = t1 - t0
print("TOOK ", dt, f" seconds {dt/L/N*fs*60*60} per hour of audio")

scat.feature_vector(res)

print(res.keys())
for p, it in res.items(): print(p, it[1].shape)


M = min(res[(0,)][0].shape[1], 15) + 1

scat.tf_filter_banks[0].plot()

plt.subplot(M//3, M//(M//3)+1, 1)
plt.imshow(res[(-1,)][0][0,0,:,:].cpu(), extent=(0,1,0,1))


for m in range(M-1):
    plt.subplot(M//3, M//(M//3)+1, m+2)
    plt.imshow(res[(0,)][1][0,m,:,:].cpu(), extent=(0,1,0,1), interpolation='nearest')
    fct, fcqf = scat.get_slope(0, m)
    plt.title("{:.2f} deg ({:.2f}, {:.2f})".format(np.rad2deg(np.arctan2(fct, fcqf)), fct, fcqf))


plt.show(block=True)
torch.cuda.empty_cache()



    