import torch
import torch.nn as nn
from torch import Tensor
from math import log2, ceil, floor
from torch.nn import Conv1d
from typing import List

from . import scattering_1d as s1d

#https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method

def print_mb(x: Tensor):
    print(x.element_size()*x.numel()/1024/1024)


class Conv1D:    
    def __init__(self, filter_weights: Tensor, N, ds =1, conv_dim = -1) -> None:
        """Create a 1D convolution layer. Internally, it may result in NN or OAS convolutions.

        Args:
            filter_weights (Tensor): Filter weight the size of (Nfilt, Lfilt), Lfilt must be odd, implicity centered at n=0.
            
            ds (int, optional): Downsampling amount. Defaults to 1.
        """
        self.ds = ds
        self.Nx = N
        self.Nh = filter_weights.shape[1]
        self.Nfilt = filter_weights.shape[0]
        assert self.Nh % 2 == 1, "Only supports odd filter lengths"
        self.signal_pad = self.Nh//2 #the amount to pad on both sides of the signal        
        self.pad_filters = self.Nx - 1 #self.Nx - self.Nh + 2*self.signal_reflection_pad = self.Nx-1
        self.padded_length = self.Nx + self.Nh - 1
        filter_weights = torch.nn.functional.pad(filter_weights, [0, self.pad_filters], mode='constant', value=0.0)  
        self.H = filter_weights.swapaxes(0, 1).cuda() #(Nh, Nfilt)    
        self.conv_dim = conv_dim 
        self._precompute_H()

    def _precompute_H(self):
        dec_m = self.padded_length//2 + 1
        dec_n = dec_m//self.ds 
        self.freq_dec_start =  dec_m - dec_n    
        self.freq_dec_end =  dec_m + dec_n +1  
        self.H = torch.fft.fft(self.H, dim=0)
        self.H = torch.fft.fftshift(self.H, dim=0)
        self.H = torch.slice_copy(self.H, start = self.freq_dec_start, end = self.freq_dec_end, dim = 0).expand(1, -1, -1) #now of shape (1, Npadded, Nfilt)
    

    def _direct(self, x: Tensor, filter_idx: Tensor) -> Tensor:
        #requires a pre-padded x
        #the convolution dimension must be the last one

        H = self.H
        Nfilt = self.Nfilt
        other_dims_shape = x.shape[0:-1]

        #select the filters
        if filter_idx != None:
            H = torch.index_select(H, dim=1, index=filter_idx.cuda()) 
            Nfilt = torch.sum(filter_idx)[0]           

        #compute x fft
        Y = torch.fft.fft(x, dim=-1)
        Y = torch.fft.fftshift(Y, dim=-1)

        #reshape using views for optimal multiplications
        Y = Y.view(-1, self.padded_length, 1) #now of shape (?, Npadded, 1)
        #downsample in frequency domain
        Y = torch.slice_copy(Y, start = self.freq_dec_start, end = self.freq_dec_end, dim = -2).expand(-1, -1, Nfilt) #now of shape (?, Npadded, Nfilt)
        k = Y.shape[0]
        H = H.expand(k, -1, -1) #(?, Npadded, Nfilt)
        #compute convolution
        Y = torch.fft.ifftshift(Y * H, dim=-2)
        return torch.fft.ifft(Y, dim=-2).reshape(*other_dims_shape, -1, Nfilt)
    
    def add_padding(self, x: Tensor, pad_mode = 'reflect'):
        x = x.swapaxes(self.conv_dim, -1)
        x = torch.nn.functional.pad(x, [self.signal_pad, self.signal_pad], mode=pad_mode) 
        return x.swapaxes(self.conv_dim, -1)
    
    def remove_padding(self, x: Tensor,  filtered = True):
        start = ceil(self.Nh / self.ds) if filtered else self.signal_pad
        length = ceil(self.Nx / self.ds) if filtered else self.Nx

        if self.Nx % 2 == 1: start -= 1

        if filtered and self.conv_dim < 0: self.conv_dim -= 1
        return torch.narrow(x, start = start, length=length, dim=self.conv_dim)
        
    def convolve(self, x: Tensor, filter_idx = None) -> Tensor:
        """1D convolution across multiple dimensions and batches. 
        Batch dimension may be treated as an arbitrary data dimension.
        Will use the FFT OAS convolution.

        Args:
            x (Tensor): Input tensor of shape (...xd)
            filter_idx (List[int], optional): List of filter indices to compute. When None, computes all filters. Default to None (all filters).
            conv_dim (int, optional): Dimension to convolve. Defaults to -1. 

        Returns:
            Tensor: Convolved signal of shape (...xd, Nfilt)
        """

        assert x.shape[self.conv_dim] == self.padded_length, f"Requires a correctly pre-padded tensor. Requires a length {self.padded_length} but got {x.shape[conv_dim]} for dim {conv_dim}."

        conv_dim = len(x.shape) + self.conv_dim if self.conv_dim < 0 else self.conv_dim
        is_last_dim = conv_dim == len(x.shape) - 1
        #swap out dimensions, since self._conv requires the last dimension
        if not is_last_dim:
            x = x.swapaxes(-1, conv_dim)           
        
        y = self._direct(x, filter_idx)
            
        #swap dimension back
        if not is_last_dim:
            y = y.swapaxes(-2, conv_dim)
        return y
    


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
        self.H = torch.fft.fftshift(self.H, dim=0)
        self.H = torch.slice_copy(self.H, start = self.freq_dec_start, end = self.freq_dec_end, dim = 0).expand(1, -1, -1) #now of shape (1, Npadded, Nfilt)


class SeperableConv:    
   
    def __init__(self, filter_weights: List[Tensor], N: List[int], ds: List[int], conv_dim: List[int]):
        self.filters: List[Filter] = []
        for h, n, d, c in zip(filter_weights, N, ds, conv_dim):
            self.filters.append(Filter(h, n, d, c))

    def _compute_fft_conv(self, x: Tensor, filter: Filter, filter_idx = None) -> Tensor:        
        """Compute the downsampled FFT convolution across the last dimension, i.e., downsample(fft(x))*H, where H has been pre-computed in the filter.

        Args:
            x (Tensor): 
            filter (FilterData): The filter with precomputed weights.
            filter_idx (List[int], optional): The filters to compute, ordered by (start, length). If None, computes all filters.

        Returns:
            Tensor: downsample(fft(x))*H
        """                 
        Nfilt = filter.Nfilt
        other_dims_shape = x.shape[0:-1]
        H = filter.H

        #select the filters
        if filter_idx != None:
            H = torch.narrow(H, dim=1, start=filter_idx[0], length=filter_idx[1]) 
            Nfilt = torch.sum(filter_idx)[0]  

        #compute x fft
        Y: Tensor = torch.fft.fft(x, dim=-1)
        Y = torch.fft.fftshift(Y, dim=-1)

        #reshape using views for optimal multiplications
        Y = Y.view(-1, filter.padded_length, 1) #now of shape (?, Npadded, 1)
        #downsample in frequency domain
        Y = torch.slice_copy(Y, start = filter.freq_dec_start, end = filter.freq_dec_end, dim = -2).expand(-1, -1, Nfilt) #now of shape (?, Npadded, Nfilt)
        k = Y.shape[0]
        H = H.expand(k, -1, -1) #(?, Npadded, Nfilt)
        #compute convolution
        Y = torch.fft.ifftshift(Y * H, dim=-2)
        return Y.reshape(*other_dims_shape, -1, Nfilt)     
  
    
    def add_padding(self, x: Tensor, pad_mode = 'reflect'):
        for f in self.filters:
            x = x.swapaxes(f.conv_dim, -1)
            x = torch.nn.functional.pad(x, [f.signal_pad, f.signal_pad + 1], mode=pad_mode) 
            x = x.swapaxes(f.conv_dim, -1)
        return x
    
    def remove_padding(self, x: Tensor,  filtered = True):
        for f in self.filters:
            start = ceil(f.Nh / f.ds) - 1 if filtered else f.signal_pad
            length = ceil(f.Nx / f.ds) if filtered else f.Nx    
            # if f.Nx % 2 == 1: start -= 1        
            x = torch.narrow(x, start = start, length=length, dim=f.conv_dim)
        return x
        
    def convolve(self, x: Tensor, filter_idx = None) -> Tensor:  
        y = x
        #compute the fft and convolution of each axis with downsampling
        for i, f in enumerate(self.filters):
            d = f.conv_dim
            y = y.swapaxes(d, -1)
            y = self._compute_fft_conv(y, f, filter_idx[i] if filter_idx else None)
            y = y.swapaxes(d, -2) #we added another dimension
        #now return the ifft
        #the returned shape is (...xd, ...xNfilt), where the additional dimensions are the filters that have been computed
        
        return torch.fft.ifftn(y, dim=[f.conv_dim for f in self.filters])
    
