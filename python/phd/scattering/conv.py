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
    def __init__(self, filter_weights: Tensor, N: int, ds: int, conv_dim: int, enable_freq_ds: bool) -> None:        
        self.ds = ds
        self.Nx = N
        self.Nh = filter_weights.shape[1]
        self.Nfilt = filter_weights.shape[0]
        assert self.Nh % 2 == 1, "Only supports odd filter lengths"
        assert conv_dim >= 0, "No negative indexing allowed for convolutions!"
        self.signal_pad = self.Nh//2     
        self.padded_length = self.Nx + self.signal_pad * 2
        self.pad_filters = self.padded_length - self.Nh
        self.raw_filter_weights = filter_weights.swapaxes(0, 1)#(Nh, Nfilt) 
        filter_weights_pad = torch.nn.functional.pad(filter_weights, [0, self.pad_filters], mode='constant', value=0.0)  
        self.H = filter_weights_pad.swapaxes(0, 1).cuda() #(Nh, Nfilt)    
        self.conv_dim = conv_dim  
        self.enable_freq_ds = enable_freq_ds       
        self._precompute_H()


    def _precompute_H(self):
        #Normal filter weights pre-computed
        self.H = torch.fft.fft(self.H, dim=0)
        if self.enable_freq_ds:
        #https://dsp.stackexchange.com/questions/64821/2d-fourier-downsampling
            dec_m = self.padded_length//2 + 1
            dec_n = dec_m//self.ds 
            #NOTE: The example uses octave, which utilizes 1-indexing and also inclusive indexing
            # we must convert to 0-index in non-inclusive end index
            self.freq_dec_start =  dec_m - dec_n -1   
            self.freq_dec_end =  dec_m + dec_n +1  -1
            self.freq_dec_len = self.freq_dec_end - self.freq_dec_start
            if self.ds == 1: 
                self.freq_dec_start = 0
                self.freq_dec_end = self.padded_length
                self.freq_dec_len = self.padded_length
            self.H = torch.fft.fftshift(self.H, dim=0)
            self.H = torch.slice_copy(self.H, start = self.freq_dec_start, end = self.freq_dec_end, dim = 0)
        


class SeperableConv:    
   
    def __init__(self, filter_weights: List[Tensor], N: List[int], ds: List[int], conv_dim: List[int], enable_freq_ds = False):
        self.filters: List[Filter] = []
        self.enable_freq_ds = enable_freq_ds
        for h, n, d, c in zip(filter_weights, N, ds, conv_dim):
            self.filters.append(Filter(h, n, d, c, enable_freq_ds))

    def _select_filters_fft(self, filter: Filter, filter_idx):
        #select the filters
        Nfilt = filter.Nfilt
        H = filter.H
        if filter_idx != None:
            H = torch.narrow(H, dim=1, start=filter_idx['start'], length=filter_idx['length']) 
            Nfilt = filter_idx['length']
        return H, Nfilt    
    
    
    def _mulacc(self, x: Tensor, y: Tensor):
        #for x, y of input dimensions (d1, d2, d3, d4)
        #compute the dot product along d3
        #output of (d1, d2, d4)
        return torch.einsum('ijkl,ijkl->ijl', (x, y))
    
    def _select_filters_straight(self, filter: Filter, filter_idx):
        #select the filters
        Nfilt = filter.Nfilt
        H = filter.raw_filter_weights
        if filter_idx != None:
            H = torch.narrow(H, dim=1, start=filter_idx['start'], length=filter_idx['length']) 
            Nfilt = filter_idx['length']
        return H, Nfilt
    
    def _apply_fun(self, x: Tensor, fun):
        if fun: 
            if fun == torch.abs: fun(x, out=x) #quick optimatisation - inplace does not require more memory
            else: x = fun(x)
        return x

    def _compute_straight_conv(self, x: Tensor, filter: Filter, filter_idx = None, fun_before_ds = None):
        other_dims_shape = x.shape[0:-1]
        H, Nfilt = self._select_filters_fft(filter, filter_idx)
        x = x.view(-1, x.shape[-1]) #now of shape (?, Nx)
        x = x.unfold(-1, filter.Nh, filter.ds) #now of shape (?, Ns, Nh), where Ns is the number of sliding blocks
        x = x.expand(-1, -1, -1, Nfilt) #(?, Ns, Nh, Nfilt)
        H = H.expand(*x.shape[0:2], -1, -1) #(?, Ns, Nh, Nfilt)
        y = self._mulacc(x, H) #(?, Ns, Nfilt)
        y = self._apply_fun(y, fun_before_ds)
        return y.reshape(*other_dims_shape, -1, Nfilt) 
    
    def _compute_fft_conv_freq_ds(self, x: Tensor, filter: Filter, filter_idx = None, fun = None) -> Tensor:        
        """Compute the downsampled FFT convolution across the last dimension, i.e., downsample(fft(x))*H, where H has been pre-computed in the filter.
        Args:
            x (Tensor): 
            filter (FilterData): The filter with precomputed weights.
            filter_idx (List[int], optional): The filters to compute, ordered by (start, length). If None, computes all filters.
        Returns:
            Tensor: downsample(fft(x))*H
        """                 
        
        other_dims_shape = x.shape[0:-1]
        H, Nfilt = self._select_filters_fft(filter, filter_idx)

        #compute x fft
        Y: Tensor = torch.fft.fft(x, dim=-1)
        Y = torch.fft.fftshift(Y, dim=-1).contiguous()

        #reshape using views for optimal multiplications
        Y = Y.view(-1, filter.padded_length, 1) #now of shape (?, Npadded, 1)
        #downsample in frequency domain
        Y = Y.expand(-1, -1, Nfilt) #now of shape (?, Npadded, Nfilt)
        Y = torch.narrow(Y, start = filter.freq_dec_start, length=filter.freq_dec_len, dim = -2) #now of shape (?, Npadded, Nfilt)
        k = Y.shape[0]
        H = H.expand(k, -1, -1) #(?, Npadded, Nfilt)
        #compute convolution
        Y = torch.fft.ifftshift(Y * H, dim=-2)
        y = torch.fft.ifft(Y, dim=-2, out=Y)
        y = self._apply_fun(y, fun)
        return y.reshape(*other_dims_shape, -1, Nfilt)     

    def _compute_fft_conv_time_ds(self, x: Tensor, filter: Filter, filter_idx = None, fun = None) -> Tensor:        
        """Compute the downsampled FFT convolution across the last dimension, i.e., downsample(fft(x))*H, where H has been pre-computed in the filter.

        Args:
            x (Tensor): 
            filter (FilterData): The filter with precomputed weights. If filter_idx is used, these filters must be in ordered. Selection is only a range, and not arbitrary locations.
            filter_idx (Dict, optional): The filters to compute, with dictionary entries of 'start' and 'length'. If None, computes all filters.
            fun_before_ds: The function to apply before downsampling in the time domain.
        Returns:
            Tensor: downsample(fft(x))*H
        """                 
        
        other_dims_shape = x.shape[0:-1]
        H, Nfilt = self._select_filters_fft(filter, filter_idx)

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
        y = torch.fft.ifft(Y, dim=-2, out = Y) 

        if filter.ds > 1: y = torch.slice_copy(y, -2, start=0, step=filter.ds)
        y = self._apply_fun(y, fun)
        return y.reshape(*other_dims_shape, -1, Nfilt)     
  
    
    def add_padding(self, x: Tensor, pad_mode = 'reflect'):
        #TODO: Add padding for when freqDS is enabled
        for f in self.filters:
            x = x.swapaxes(f.conv_dim, -1)
            x = torch.nn.functional.pad(x, [0, 2*f.signal_pad], mode=pad_mode) 
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
        y = x
        #compute the fft and convolution of each axis with downsampling
        for i, f in enumerate(self.filters):
            d = f.conv_dim
            y = y.swapaxes(d, -1)
            if self.enable_freq_ds:
                y = self._compute_fft_conv_freq_ds(y, f, filter_idx[d] if filter_idx else None, fun_before_ds)
            else:
                y = self._compute_fft_conv_time_ds(y, f, filter_idx[d] if filter_idx else None, fun_before_ds)
            y = y.swapaxes(d, -2) #we added another dimension
        #now return the ifft
        #the returned shape is (...xd, ...xNfilt), where the additional dimensions are the filters that have been computed
        
        return y
    
