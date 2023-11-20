import sys
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../python")

import phd.detector.spectral_entropy as se
from phd.scattering.scattering_1d import TORCH_OUTPUT_DATA_TYPE

Lf = 50
Lt = 10000
Nb = 10

s = torch.randn((Nb, 1, Lf, Lt), dtype=TORCH_OUTPUT_DATA_TYPE)*1.0
s = torch.abs(s)
s[:, 0, 10, (Lt//3):(2*Lt//3)] += 3.0

fs = 10
Tmf = 5

det = se.SEDetector(Tmf, fs, decimate=2)

prob = det.signal_probability(s)
H = det._entropy(s)
h = torch.linspace(3, 4, 1000)
# g = det.segmm._gauss(h[None, None, :] - det.segmm.mu).cpu()


# plt.plot(h.cpu(), g[0, 0, :])
# plt.plot(h.cpu(), g[1, 0, :])
# plt.show(block=True)

print(prob.shape, prob.dtype)

plt.subplot(211)
plt.plot(prob[0,0,:].cpu())
plt.plot(prob[1,0,:].cpu())
plt.subplot(212)
plt.plot(H[0,0,:].cpu())
plt.plot(H[1,0,:].cpu())
plt.show(block=True)

torch.cuda.empty_cache()