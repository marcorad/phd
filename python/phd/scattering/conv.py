import torch
import torch.nn as nn
from torch import Tensor
from math import log2, ceil, floor
from torch.nn import Conv1d

from . import scattering_1d as s1d

#https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method

def _block_fft_cost(m, n):
    return n * (log2(n) + 1)/(n - m + 1)

def print_mb(x: Tensor):
    print(x.element_size()*x.numel()/1024/1024)

def _optimise_cost(m):
    n = m
    cost = _block_fft_cost(m, n)
    prev_cost = cost + 1
    while cost - prev_cost < 0:
        prev_cost = cost
        n += 1
        cost = _block_fft_cost(m, n) 
        
    n-=1
    return 2**(floor(log2(n)))

class Conv1D:    
    def __init__(self, filter_weights: Tensor, ds =1) -> None:
        """Create a 1D convolution layer. Internally, it may result in NN or OAS convolutions.

        Args:
            filter_weights (Tensor): Filter weight the size of (Nfilt, Lfilt), Lfilt must be odd, implicity centered at n=0.
            
            ds (int, optional): Downsampling amount. Defaults to 1.
        """
        self.ds = ds
        lenH = filter_weights.shape[-1]
        self.M = 2**ceil(log2(lenH))
        self.lenH = lenH    
        self.Nir = _optimise_cost(self.M)
        pad = [0, self.Nir - lenH]
        self.ir_pad = pad[1]
        self.Nfilt = filter_weights.shape[0]
        filter_weights = nn.functional.pad(filter_weights, pad, mode='constant', value=0.0)        
        self.H = torch.fft.fft(filter_weights.cuda(), dim=-1) #precompute filter FFT
        self.overlap = self.M-1
        self.step_size = self.Nir - self.overlap 
        
    
        
    def _conv(self, x: Tensor, filter_idx = None):     
         
        #requires the last dimension to be the convolution dimension

        origL = x.shape[-1]

        #TODO: Do normal FFT conv for short signals, i.e. when len(padded x) < Nir

        #input shape of (...xd)
        #x reshaped to (...x(d-1), 1, #windows, Nir)
        #filter shape of (1...x(d-1), Nfilt, 1, Nir)

        #reshape filter stored as (Nfilt, Nir)
        H = self.H[:, None, :] #(Nfilt, 1, Nir)
        d = len(x.shape)
        for _ in range(d - 1): H = torch.unsqueeze(H, 0) #(1...x(d-1), Nfilt, 1, Nir)


        #select the filters
        if filter_idx != None:
            H = torch.index_select(H, dim=-3, index=filter_idx)              

        
        P = self.lenH//2 
        Pend = self.Nir*(ceil((origL+2*P + self.M)/self.Nir)+1) - (origL+2*P + self.M)         
        x = nn.functional.pad(x, [P + self.M, P + Pend], mode='constant', value=0.0)
        x = x.unfold(-1, self.Nir, self.step_size) #(...x(d-1), #windows, Nw)
        x = torch.unsqueeze(x, -3) #(...x(d-1), 1, #windows, Nw)
        x = x.cuda()
        Y: Tensor = torch.fft.fft(x, dim=-1)            

        Y = Y*H#(1...x(d-1), Nfilt, 1, Nw)*(...x(d-1), 1, #windows, Nir)=(...x(d-1), Nfilt, #windows, Nir)
        del H   

        Y = torch.fft.ifft(Y, dim=-1)       
        
        Y = torch.slice_copy(Y, -1, start=self.M-1, end=self.Nir) #discard irrelevant parts  
        
        s = list(Y.shape[:-1])
        s[-1] = s[-1]*Y.shape[-1] 
        Y = Y.reshape(s)  #concatenate windows (...x(d-1), Nfilt, d[-1] + pad)
        Y = torch.slice_copy(Y, -1, start=self.lenH, end=origL + self.lenH, step=self.ds) #trim IR ends and downsample (...x(d-1), Nfilt, downsample(d[-1]))
        
        return torch.swapaxes(Y, -1, -2) #(...xd, Nfilt)
        
        
    #input Tensor of size (Nbatch, 1, Nch, Nx)
    def convolve(self, x: Tensor, filter_idx = None, conv_dim = -1) -> Tensor:
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
        #swap out dimensions, since self._conv requires the last dimension
        if conv_dim != -1:
            x = x.swapaxes(-1, conv_dim)           
        
        y = self._conv(x, filter_idx)
            
        #swap dimension back
        if conv_dim != -1:
            y = y.swapaxes(-2, conv_dim-1) #extra dimension added
        
        return y
    
    


            
            
        
        
    