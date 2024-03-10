# from medmnist import OrganMNIST3D, FractureMNIST3D, NoduleMNIST3D, AdrenalMNIST3D, VesselMNIST3D, SynapseMNIST3D 


# data = OrganMNIST3D(split="test", download=True, root='/media/data/Datasets/organ-mnist-3d/')
# data = FractureMNIST3D(split="test", download=True, root='/media/data/Datasets/organ-mnist-3d/')
# data = AdrenalMNIST3D(split="test", download=True, root='/media/data/Datasets/organ-mnist-3d/')
# data = NoduleMNIST3D(split="test", download=True, root='/media/data/Datasets/organ-mnist-3d/')
# data = VesselMNIST3D(split="test", download=True, root='/media/data/Datasets/organ-mnist-3d/')
# data = SynapseMNIST3D(split="test", download=True, root='/media/data/Datasets/organ-mnist-3d/')

# exit()

import numpy as np

import matplotlib.pyplot as plt

import sys
sys.path.append('../python')
from phd.dataprocessing.medmnist3d import load_train_test
# import phd.scattering.config as config
# # config.MORLET_DEFINITION = config.MorletDefinition(2,4,2,1,2)
# config.MORLET_DEFINITION = config.MorletDefinition(2,3,2,1,2)
# config.set_precision('single')
# config.ENABLE_DS = True
# from phd.scattering.sep_ws import optimise_T, SeperableWaveletScattering


# import torch

# Q = [[1]*3]*1
# T = [optimise_T(32,1)]*3
# print(T)
# fs = [1, 1, 1]
# dims = [1, 2, 3]
# N = [28, 28, 28]

# x = X_train[0:64, :, :, :].astype(np.float32)/255
# print(x.dtype)

# ws = SeperableWaveletScattering(Q, T, fs, dims, N)
# s = ws.scatteringTransform(torch.from_numpy(x), batch_dim=0, batch_size=8)

# print(s.shape)

X_train, y_train, _, _ = load_train_test('fracture', True)
idx = 120
x = X_train[idx, :, :, :]
print(y_train[idx])

for i in range(28):
    plt.subplot(6, 6, i+1)
    plt.imshow(x[i, :, :])
    
plt.show(block=True)