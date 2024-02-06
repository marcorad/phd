from phd.dataprocessing.esc50 import load_esc50

from phd.scattering import config
config.ENABLE_FREQ_DS = False
config.PAD_MODE = 'constant'

from phd.scattering.sep_ws import JointTFScattering, optimise_T

import pandas as pd
import torch
import pickle as pkl

audio, labels = load_esc50()
fs = 14700
fstart = 50

df = pd.DataFrame(labels)

T = [optimise_T(1, fs), optimise_T(32, 1)]
print("T ", T)
Tstr = str(T).replace(".", "")
print(Tstr)
Q = [4, [2, 1], [1, 1]]
Qstr = "-".join([str(q) for q in Q])

ws = JointTFScattering(Q, T, fs, dim=1, N=audio.shape[1], fstart=fstart)

x = torch.from_numpy(audio)

s = ws.scatteringTransform(x, batch_dim=0, batch_size=1)

with open(f'data/ws-esc50-{Tstr}-{fstart}-{Qstr}.pkl', 'wb') as file:
    pkl.dump(s, file)

