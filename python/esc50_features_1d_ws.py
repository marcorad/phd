from sepws.dataprocessing.esc50 import load_esc50

from sepws.scattering import config
config.ENABLE_FREQ_DS = False
config.PAD_MODE = 'reflect'

from sepws.scattering.sep_ws import SeperableWaveletScattering, optimise_T

import pandas as pd
import torch
import pickle as pkl

audio, labels = load_esc50()
fs = 8000
fstart = 50

df = pd.DataFrame(labels)

T = optimise_T(2, fs)
print("T ", T)
Tstr = str(T).replace(".", "")
print(Tstr)
Q = [8, 2, 1]
Qstr = "-".join([str(q) for q in Q])

ws = SeperableWaveletScattering([[q] for q in Q], [T], [fs], dims=[1], N=[audio.shape[1]], fstart=[fstart])

x = torch.from_numpy(audio)

s = ws.scatteringTransform(x, batch_dim=0, batch_size=16)

with open(f'data/ws-esc50-{Tstr}-{fstart}-{Qstr}.pkl', 'wb') as file:
    pkl.dump(s, file)

