import torch
import torch.nn as nn
from torch import Tensor
from math import log2, ceil, floor
from torch.nn import Conv1d

from . import scattering_1d as s1d

#https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method

def _block_fft_cost(m, n):
    return n * (log2(n) + 1)/(n - m + 1)

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
    def __init__(self, filter_weights: Tensor, conv_dim = -1, ds =1) -> None:
        """Create a 1D convolution layer. Internally, it may result in NN or OAS convolutions.

        Args:
            filter_weights (Tensor): Filter weight the size of (Nfilt, Lfilt), Lfilt must be odd, implicity centered at n=0.
            conv_dim (int, optional): Dimension to convolve. Defaults to -1.
            ds (int, optional): Downsampling amount. Defaults to 1.
        """
        #filter weights of of size (Nfilt, Lfilt)
        self.conv_dim = conv_dim
        self.ds = ds
        lenH = filter_weights.shape[-1]
        self.M = 2**ceil(log2(lenH))
        self.lenH = lenH
        self.N = _optimise_cost(self.M)
        pad = [0, self.N - lenH]
        self.ir_pad = pad[1]
        self._init_straight_conv_layer(filter_weights)
        self.Nfilt = filter_weights.shape[0]
        filter_weights = nn.functional.pad(filter_weights, pad, mode='constant', value=0.0)        
        self.H = torch.fft.fft(filter_weights.cuda(), dim=-1) #precompute filter FFT
        self.H = self.H[None, :, None, None, :] #(1, Nfilt, 1, 1, Nh)
        self.overlap = self.M-1
        self.step_size = self.N - self.overlap
        
        
    def _init_straight_conv_layer(self, w: Tensor):
        
        self.conv_layer = Conv1d(
            in_channels=1,
            out_channels=w.shape[0],
            kernel_size=w.shape[1],
            padding=w.shape[1]//2,
            bias=False,
            dtype=s1d.TORCH_DATA_TYPE,
            stride=self.ds
        )
        self.conv_layer.weight = nn.Parameter(w[:, None, :], requires_grad=False)
    
    #input Tensor of size (Nbatch, 1, Nch, Nx)
    def _conv_straight(self, x: Tensor):
        print("USING STRAIGHT CONV")
        y = torch.zeros((x.shape[0], self.Nfilt, x.shape[2], ceil(x.shape[3]/self.ds)), dtype=s1d.TORCH_DATA_TYPE)
        for i in range(x.shape[-2]):
            y[:, :, i, :] = self.conv_layer(x[:, :, i, :])
        return y
        
    def _conv_oas(self, x: Tensor, filter_idx = None):     
        # print("USING OAS CONV")   
        #requires the last dimension to be the convolution dimension

        #compute OAS windows for x
        origL = x.shape[-1]
        P = self.lenH//2 
        Pend = self.N*(ceil((origL+2*P + self.M)/self.N)+1) - (origL+2*P + self.M)         
        x = nn.functional.pad(x, [P + self.M, P + Pend], mode='constant', value=0.0)
        x = x.unfold(-1, self.N, self.step_size) #(Nbatch, 1, Nch, Nwind, Nh)
        x = x.cuda()
        Y: Tensor = torch.fft.fft(x, dim=-1)

        #select the filters
        H = self.H if filter_idx == None else self.H[:, filter_idx, :, :, :]

        Y = Y*H #(Nbatch, 1, Nch, Nwind, Nh) * (1, Nfilt, 1, 1, Nh)
        Y = torch.fft.ifft(Y, dim=-1)
        Y = Y[:, :, :, :, (self.M-1):self.N] #OAS segments        
        s = list(Y.shape[:-1])
        s[-1] = s[-1]*Y.shape[-1]
        Y = Y.reshape(s) 
        Y = Y[:, :, :, self.lenH:(origL + self.lenH):self.ds]     
           
        return Y #(Nbatch, Nfilt, Nch, Nx)
    
    def _use_oas(self, N):
        return self.ds < (self.M-1)/(2*log2(self.M) + 1) and self.M < N
        
    #input Tensor of size (Nbatch, 1, Nch, Nx)
    def conv_multiple(self, x: Tensor, filter_idx = None) -> Tensor:
        """1D convolution across multiple dimensions and batches. 
        Batch dimension may be treated as an arbitrary data dimension.
        Will use the FFT OAS convolution.

        Args:
            x (Tensor): Input tensor of shape (Nbatch, 1, Nch, Nx)
            filter_idx (List[int], optional): List of filter indices to compute. When None, computes all filters. Default to None (all filters).

        Returns:
            Tensor: Convolved signal of shape (Nbatch, Nfilt, Nch, Nx)
        """
        #swap out dimensions, since convolution requires the last dimension
        if self.conv_dim != -1:
            x = x.swapaxes(-1, self.conv_dim)
            
        #convolve optimally
        # DISABLE NN CONV FOR NOW
        # if self._use_oas(x.shape[self.conv_dim]):
        #     y = self._conv_oas(x)
        # else:
        #     y =  self._conv_straight(x)
            
        y = self._conv_oas(x, filter_idx)
            
        #swap dimension back
        if self.conv_dim != -1:
            y = y.swapaxes(-1, self.conv_dim)

        torch.cuda.empty_cache()
        return y
    
    #input Tensor of size (Nbatch, 1, Nx)
    def conv(self, x: Tensor):
        
        return self.conv_multiple(x[:, :, None, :])[:, :, 0, :] #(Nbatch, Nfilt, Nx)
        



            
            
        
        
    