import torch
from torch import pi
from ..scattering_1d import TORCH_DATA_TYPE

def _fm(fs, N, flo, fm, df, Nbatch, func):
    
    n = torch.arange(N)
    t = n / fs
    dw = torch.cumsum(2*pi*flo + df*2*pi*(func(t*fm)), 0) / fs
    x = torch.zeros((Nbatch, 1, N), dtype=TORCH_DATA_TYPE)
    x[:, 0, :] = torch.sin(dw)
    return x
    
def _tri(t):
    return 2*torch.abs(2*(t - torch.floor(t + 0.5))) - 1

def _sin(t):
    return torch.sin(2*pi*t)

def fm_sin(fs, N, flo, fm, df, Nbatch=1):
    return _fm(fs, N, flo, fm, df, Nbatch, _sin)

def fm_tri(fs, N, flo, fm, df, Nbatch=1):
    return _fm(fs, N, flo, fm, df, Nbatch, _tri)