from sepws.dataprocessing.esc50 import load_esc50

from sepws.scattering import config
config.PAD_MODE = 'constant'
config.set_precision('single')
config.MORLET_DEFINITION = config.MorletDefinition(2,3,2,2,3)

from sepws.scattering.sep_ws import JointTFScattering, optimise_T, SeperableWaveletScattering

import pandas as pd
import torch
import pickle as pkl

X_train, y_train, X_test, y_test = load_esc50()
fs = 44100/3
fstart = 20



Tt = optimise_T(2048/fs, fs, eps=0.1)
Ttf = optimise_T(32, 1)
T = [Tt, Ttf]
Qt = 8
Qtf = [[1, 1], [1, 1]]
Q = [Qt, Qtf]

ws = JointTFScattering(Qt, Qtf,  Tt, Ttf, fs, dim=1, N=X_train.shape[1], fstart=fstart)

# T = [optimise_T(4096/fs, fs, eps=0.1)]
# Q = [[8], [4], [1]]
# ws = SeperableWaveletScattering(Q, T, [fs], dims=[1], N=[X_train.shape[1]], fstart=[fstart])

X_train = torch.from_numpy(X_train)
S_train = ws.scatteringTransform(X_train, batch_dim=0, batch_size=64, log_dct=True).numpy()
X_test = torch.from_numpy(X_test)
S_test = ws.scatteringTransform(X_test, batch_dim=0, batch_size=64, log_dct=True).numpy()


Qstr = "-".join([str(q) for q in Q])
Tstr = str(T).replace(".", "")

with open(f'data/ws-esc50-{Tstr}-{fstart}-{Qstr}.pkl', 'wb') as file:
    pkl.dump((S_train, y_train, S_test, y_test), file)

