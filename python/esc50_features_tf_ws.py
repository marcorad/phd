from phd.dataprocessing.esc50 import load_esc50

from phd.scattering import config
config.PAD_MODE = 'constant'
config.set_precision('single')

from phd.scattering.sep_ws import JointTFScattering, optimise_T

import pandas as pd
import torch
import pickle as pkl

audio, labels = load_esc50()
fs = 14700
fstart = 20

df = pd.DataFrame(labels)

Tt = optimise_T(0.5, fs)
Ttf = optimise_T(16, 1)
T = [Tt, Ttf]
print("T ", T)
Tstr = str(T).replace(".", "")
print(Tstr)


Qt = 8
Qtf = [[1, 1], [1, 1]]
Q = [Qt, Qtf]
Qstr = "-".join([str(q) for q in Q])


ws = JointTFScattering(Qt, Qtf,  Tt, Ttf, fs, dim=1, N=audio.shape[1], fstart=fstart)


x = torch.from_numpy(audio)
print(x.device)
print(f'MEMORY USAGE: {torch.cuda.memory_allocated()/1024/1024/1024} GB')

s = ws.scatteringTransform(x, batch_dim=0, batch_size=4)

with open(f'data/ws-esc50-{Tstr}-{fstart}-{Qstr}.pkl', 'wb') as file:
    pkl.dump(s, file)

