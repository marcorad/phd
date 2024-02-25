import torch
from torch import Tensor
from math import log2, ceil, floor
from typing import List, Dict


class Filter:
    def __init__(self, filter_weights: Tensor, N: int, ds: int, conv_dim: int) -> None:        
        self.ds = ds
        self.Nx = N
        self.Nh = filter_weights.shape[0]
        assert self.Nh % 2 == 1, "Only supports odd filter lengths"
        assert conv_dim >= 0, "No negative indexing allowed for convolutions!"
        self.signal_pad = self.Nh//2     
        self.padded_length = self.Nx + self.signal_pad * 2
        self.pad_filters = self.padded_length - self.Nh
        self.raw_filter_weights = filter_weights
        
        
        self.freq_ds_pad = self.ds - (self.padded_length % self.ds)
        
        filter_weights_pad = torch.nn.functional.pad(filter_weights, [0, self.pad_filters + self.freq_ds_pad], mode='constant', value=0.0)  
        self.H = filter_weights_pad.cuda() #(Nh,)    
        self.conv_dim = conv_dim  
        self.Nx_out = int(ceil(self.Nx / self.ds))
        self.unpad_start = int(ceil(self.signal_pad / self.ds))
        self._precompute_H()

    def _precompute_H(self):
        #Normal filter weights pre-computed
        self.H = torch.fft.fft(self.H, dim=0)          


class SeperableConv:    
   
    def __init__(self, filter_weights: List[Tensor], N: List[int], ds: List[int], conv_dim: List[int]):
        self.filters: List[Filter] = []
        for h, n, d, c in zip(filter_weights, N, ds, conv_dim):
            self.filters.append(Filter(h, n, d, c))
        self.output_length = []
        for f in self.filters:
            length = ceil(f.Nx / f.ds)
            self.output_length.append(length)  
   
    
    def _apply_fun(self, x: Tensor, fun):
        if fun: 
            if fun == torch.abs: x = fun(x, out=x) #quick optimatisation - inplace does not require more memory
            else: x = fun(x)
        return torch.real(x)
    
    def _freq_subsample(self, X: Tensor, ds: int):
        if X.shape[0] > 1:
            X = X.view(X.shape[0], ds, -1)
            return X.mean(axis=1, keepdim=False)   
        else:
            X = X.view(ds, -1)
            return X.mean(axis=0, keepdim=False)         
    

    def _compute_fft_conv(self, Y: Tensor, filter: Filter, fun = None) -> Tensor:    
        
        other_dims_shape = Y.shape[0:-1]
        H = filter.H        

        #reshape using views for optimal multiplications
        Y = Y.view(-1, filter.padded_length + filter.freq_ds_pad) #now of shape (?, Npadded)                
        
        k = Y.shape[0]
        H = H.expand(k, -1) #(?, Npadded)
        #compute convolution        
        Y = Y * H
        Y = self._freq_subsample(Y, filter.ds)
        y: Tensor = torch.fft.ifft(Y, dim=-1, out = Y) 

        # if filter.ds > 1: y = torch.slice_copy(y, -1, start=0, step=filter.ds)
        # y = self._apply_fun(y, fun)
        return y.reshape(*other_dims_shape, -1) 
    
    def fft(self, x: Tensor) -> Tensor:
        pad = [f.padded_length + f.freq_ds_pad for f in self.filters]
        dims = [f.conv_dim for f in self.filters]
        return torch.fft.fftn(x, s=pad, dim=dims)
    
    def _unpad(self, x: Tensor, filter: Filter) -> Tensor:
        return x.narrow(dim=-1, start=filter.unpad_start, length=filter.Nx_out)    
        
    def convolve(self, Y: Tensor, fun_before_ds = None) -> Tensor:          
        for i, f in enumerate(self.filters):
            d = f.conv_dim
            Y = Y.swapaxes(d, -1)
            Y = self._compute_fft_conv(Y, f, fun_before_ds)  
            Y = self._unpad(Y, f)          
            Y = Y.swapaxes(d, -1) 
        Y = self._apply_fun(Y, fun_before_ds) 
        return Y
   
import sys

sys.path.append('../python')

from phd.scattering.morlet import sample_gauss

import numpy as np

import matplotlib.pyplot as plt


#1D test
Nx = 32
fs = 1
sigma = 2
Nh = 8
t = np.arange(-Nh, Nh+1, dtype=np.float32)/fs    
    
h = sample_gauss(t, sigma)

x1 = np.zeros((Nx,), dtype=np.float32)
x1[Nx//2] = 1.0

# plt.subplot(211)
# plt.plot(t, h)
# plt.subplot(212)
# plt.plot(x1)
# plt.show()

conv = SeperableConv([torch.from_numpy(h)], [Nx], [8], [0])
Y = conv.fft(torch.from_numpy(x1).cuda())

y = conv.convolve(Y, torch.real).cpu().numpy()
plt.plot(y)
plt.show()

#2D test
Nx = 32
fs = 1
sigma = 2
Nh = 8
t = np.arange(-Nh, Nh+1, dtype=np.float32)/fs    
    
h = sample_gauss(t, sigma)

x1 = np.zeros((Nx,Nx), dtype=np.float32)
x1[Nx//2, Nx//2] = 1.0

# plt.subplot(211)
# plt.plot(t, h)
# plt.subplot(212)
# plt.plot(x1)
# plt.show()

h_torch = torch.from_numpy(h)

conv = SeperableConv([h_torch, h_torch], [Nx, Nx], [2, 2], [0, 1])
Y = conv.fft(torch.from_numpy(x1).cuda())

y = conv.convolve(Y, torch.real).cpu().numpy()
plt.imshow(y)
plt.show()

    