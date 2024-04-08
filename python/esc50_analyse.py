from sepws.scattering import config
config.ENABLE_DS = True
config.ENABLE_FREQ_DS = False

from sepws.dataprocessing.esc50 import load_esc50, read_meta


from sepws.scattering.sep_ws import SeperableWaveletScattering, optimise_T

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

audio, labels = load_esc50()
meta = read_meta()
fs = 8860
fstart = 50

df = pd.DataFrame(labels)

print(audio.shape)

print(df.value_counts())

T = optimise_T(0.25, fs)
print("T ", T)
Tstr = str(T).replace(".", "")
print(Tstr)
Q = [8]
Qstr = "-".join([str(q) for q in Q])

ws = SeperableWaveletScattering([[q] for q in Q], [T], [fs], dims=[1], N=[audio.shape[1]], fstart=[fstart], include_on_axis_wavelets=False)

i = 90

x = torch.from_numpy(audio[[i], :])
print(x.shape)

U, s = ws.scatteringTransform(x, discard_U=False)

u = torch.log(U[0].cpu())

print(u.shape)

plt.imshow(u[0, :, :].T.flipud(), extent=(ws.sws_layers[0].samplers[0].fc[0], ws.sws_layers[0].samplers[0].fc[-1], ws.sws_layers[0].samplers[0].fc[0], ws.sws_layers[0].samplers[0].fc[-1]))
plt.title(labels[i] + f' ({list(meta.keys())[i]})')
plt.show()

Np = 8
psi = ws.sws_layers[0].samplers[0].psi
print(psi.shape)

for i in range(psi.shape[0]):
    plt.subplot(Np, Np, i + 1)
    plt.plot(np.abs(np.fft.fft(psi[i, :])))
    
plt.show()


