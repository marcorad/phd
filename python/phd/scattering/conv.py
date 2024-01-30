import torch
import torch.nn as nn
from torch import Tensor
from math import log2, ceil, floor
from torch.nn import Conv1d
from typing import List, Dict

from . import scattering_1d as s1d

#https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method

def print_mb(x: Tensor):
    print(x.element_size()*x.numel()/1024/1024)    


class Filter:
    def __init__(self, filter_weights: Tensor, N: int, ds: int, conv_dim: int) -> None:        
        self.ds = ds
        self.Nx = N
        self.Nh = filter_weights.shape[1]
        self.Nfilt = filter_weights.shape[0]
        assert self.Nh % 2 == 1, "Only supports odd filter lengths"
        assert conv_dim >= 0, "No negative indexing allowed for convolutions!"
        self.signal_pad = self.Nh//2 #the amount to pad on both sides of the signal        
        self.pad_filters = self.Nx - 1 + 1 #self.Nx - self.Nh + 2*self.signal_reflection_pad = self.Nx-1
        self.padded_length = self.Nx + self.Nh - 1 + 1
        filter_weights = torch.nn.functional.pad(filter_weights, [0, self.pad_filters], mode='constant', value=0.0)  
        self.H = filter_weights.swapaxes(0, 1).cuda() #(Nh, Nfilt)    
        self.conv_dim = conv_dim 
        self._precompute_H()

    def _precompute_H(self):
        #https://dsp.stackexchange.com/questions/64821/2d-fourier-downsampling
        dec_m = self.padded_length//2 + 1
        dec_n = dec_m//self.ds 
        #NOTE: The example uses octave, which utilizes 1-indexing and also inclusive indexing
        # we must convert to 0-index in non-inclusive end index
        self.freq_dec_start =  dec_m - dec_n -1   
        self.freq_dec_end =  dec_m + dec_n +1  -1
        self.H = torch.fft.fft(self.H, dim=0)
        


class SeperableConv:    
   
    def __init__(self, filter_weights: List[Tensor], N: List[int], ds: List[int], conv_dim: List[int]):
        self.filters: List[Filter] = []
        for h, n, d, c in zip(filter_weights, N, ds, conv_dim):
            self.filters.append(Filter(h, n, d, c))

    def _compute_fft_conv(self, x: Tensor, filter: Filter, filter_idx = None, fun_before_ds = None) -> Tensor:        
        """Compute the downsampled FFT convolution across the last dimension, i.e., downsample(fft(x))*H, where H has been pre-computed in the filter.

        Args:
            x (Tensor): 
            filter (FilterData): The filter with precomputed weights. If filter_idx is used, these filters must be in ordered. Selection is only a range, and not arbitrary locations.
            filter_idx (Dict, optional): The filters to compute, with dictionary entries of 'start' and 'length'. If None, computes all filters.
            fun_before_ds: The function to apply before downsampling in the time domain.
        Returns:
            Tensor: downsample(fft(x))*H
        """                 
        Nfilt = filter.Nfilt
        other_dims_shape = x.shape[0:-1]
        H = filter.H

        #select the filters
        if filter_idx != None:
            H = torch.narrow(H, dim=1, start=filter_idx['start'], length=filter_idx['length']) 
            Nfilt = filter_idx['length']

        #compute x fft
        Y: Tensor = torch.fft.fft(x, dim=-1).contiguous()

        #reshape using views for optimal multiplications
        Y = Y.view(-1, filter.padded_length, 1) #now of shape (?, Npadded, 1)
        #downsample in frequency domain
        Y = Y.expand(-1, -1, Nfilt) #now of shape (?, Npadded, Nfilt)
        k = Y.shape[0]
        H = H.expand(k, -1, -1) #(?, Npadded, Nfilt)
        #compute convolution
        Y = Y * H
        y = torch.fft.ifft(Y, dim=-2)
        if fun_before_ds: y = fun_before_ds(y)
        if filter.ds > 1: y = torch.slice_copy(y, -2, start=0, step=filter.ds)
        return y.reshape(*other_dims_shape, -1, Nfilt)     
  
    
    def add_padding(self, x: Tensor, pad_mode = 'reflect'):
        for f in self.filters:
            x = x.swapaxes(f.conv_dim, -1)
            x = torch.nn.functional.pad(x, [0, 2*f.signal_pad + 1], mode=pad_mode) 
            x = x.swapaxes(f.conv_dim, -1)
        return x
    
    def remove_padding(self, x: Tensor,  filtered = True):
        for f in self.filters:
            start = ceil(f.Nh / f.ds / 2) - 1 if filtered else f.signal_pad
            length = ceil(f.Nx / f.ds) if filtered else f.Nx    
            # if f.Nx % 2 == 1: start -= 1        
            x = torch.narrow(x, start = start, length=length, dim=f.conv_dim)
        return x
        
    def convolve(self, x: Tensor, filter_idx = None, fun_before_ds = None) -> Tensor:  
        y = x.cuda()
        #compute the fft and convolution of each axis with downsampling
        for i, f in enumerate(self.filters):
            d = f.conv_dim
            y = y.swapaxes(d, -1)
            y = self._compute_fft_conv(y, f, filter_idx[d] if filter_idx else None, fun_before_ds)
            y = y.swapaxes(d, -2) #we added another dimension
        #now return the ifft
        #the returned shape is (...xd, ...xNfilt), where the additional dimensions are the filters that have been computed
        
        return y
    
