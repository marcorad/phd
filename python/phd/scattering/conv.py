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
        self.M = lenH #2**ceil(log2(lenH))
        self.lenH = lenH    
        self.Nir = _optimise_cost(self.M)
        pad = [0, self.Nir - lenH]
        self.ir_pad = pad[1]
        self.Nfilt = filter_weights.shape[0]

        #filter for OAS
        h = nn.functional.pad(filter_weights, pad, mode='constant', value=0.0)        
        self.H_oas = torch.fft.fft(h.cuda(), dim=-1) #precompute filter FFT
        self.overlap = self.M-1
        self.step_size = self.Nir - self.overlap 

        self.sig_len_limit = ceil(2**((log2(self.Nir)+1) * self.Nir / (self.Nir - self.lenH + 1) -1))

        #filter for direct FFT convolution        
        self.h_direct: Tensor = filter_weights #precompute filter FFT
        self.prev_direct_size = None
        self.prev_direct_H = None


    def _oas(self, x: Tensor, filter_idx):
        # print("OAS")
        #requires the last dimension to be the convolution dimension

        origL = x.shape[-1]        

        #input shape of (...xd)
        #x reshaped to (...x(d-1), 1, #windows, Nir)
        #filter shape of (1...x(d-1), Nfilt, 1, Nir)

        #select the filters
        if filter_idx != None:
            H = torch.index_select(H, dim=0, index=filter_idx)  

        #reshape filter stored as (Nfilt, Nir)
        H = self.H_oas[:, None, :] #(Nfilt, 1, Nir)
        d = len(x.shape)
        for _ in range(d - 1): H = torch.unsqueeze(H, 0) #(1...x(d-1), Nfilt, 1, Nir)


                       
        
        
        P = self.lenH//2 
        Pend = self.Nir*(ceil((origL+2*P + self.M)/self.Nir)+1) - (origL+2*P + self.M)         
        x = nn.functional.pad(x, [P + self.M, P + Pend], mode='constant', value=0.0)
        x = x.unfold(-1, self.Nir, self.step_size) #(...x(d-1), #windows, Nw)
        x = torch.unsqueeze(x, -3) #(...x(d-1), 1, #windows, Nw)
        x = x.cuda()
        Y: Tensor = torch.fft.fft(x, dim=-1)  
        torch.cuda.empty_cache()              

        Y = Y*H#(1...x(d-1), Nfilt, 1, Nw)*(...x(d-1), 1, #windows, Nir)=(...x(d-1), Nfilt, #windows, Nir)
        del H   

        Y = torch.fft.ifft(Y, dim=-1)       
        
        Y = torch.slice_copy(Y, -1, start=self.M-1, end=self.Nir) #discard irrelevant parts  
        
        s = list(Y.shape[:-1])
        s[-1] = s[-1]*Y.shape[-1] 
        Y = Y.reshape(s)  #concatenate windows (...x(d-1), Nfilt, d[-1] + pad)
        Y = torch.slice_copy(Y, -1, start=self.lenH, end=origL + self.lenH, step=self.ds) #trim IR ends and downsample (...x(d-1), Nfilt, downsample(d[-1]))
        
        return torch.swapaxes(Y, -1, -2) #(...xd, Nfilt)
    

    def _direct(self, x: Tensor, filter_idx):

        #TODO: do subsampling in frequency domain

        # print("DIRECT")
        origL = x.shape[-1]

        H = self.h_direct #(Nfilt, Nh)

            
        pad_len = origL + self.lenH - 1
        #https://dsp.stackexchange.com/questions/64821/2d-fourier-downsampling
        dec_m= pad_len//2 + 1
        dec_n=dec_m//self.ds

        if pad_len == self.prev_direct_size:
            H = self.prev_direct_H
        else:
            H = nn.functional.pad(H, [0, (origL + self.lenH - 1) - self.lenH], mode='constant', value=0.0)
            H = H.swapaxes(0, 1).cuda() #(Nh, Nfilt)
            H = torch.fft.fft(H, dim=0)
            H = torch.fft.fftshift(H, dim=0)
            H = torch.slice_copy(H, start=dec_m - dec_n, end=dec_m + dec_n +1, dim=0)
            self.prev_direct_H = H
            self.prev_direct_size = pad_len


        #select the filters
        if filter_idx != None:
            H = torch.index_select(H, dim=1, index=filter_idx.cuda())            

        d = len(x.shape)
        for _ in range(d-1): H = torch.unsqueeze(H, 0) #(1...x(d-1), Nir, Nfilt)

         
            

        #pad x to the max signal length of efficiency
        P = self.lenH//2   
        x = nn.functional.pad(x, [P, P], mode='constant', value=0.0)
        x = x.unsqueeze(-1).cuda() #(...x(d-1), Nir, 1)

        Y: Tensor = torch.fft.fft(x, dim=-2)
        Y = torch.fft.fftshift(Y, dim=-2)
        Y = torch.slice_copy(Y, start=dec_m - dec_n, end=dec_m + dec_n +1, dim=-2)
        del x
        Y = H * Y
        

        

        start = self.lenH - 1

        Y = torch.fft.ifft(torch.fft.ifftshift(Y, dim=-2), dim=-2)
        # Y = torch.slice_copy(Y, dim=-2, start=start, end=start+origL, step=self.ds)
        Y = torch.slice_copy(Y, dim=-2, start=start//self.ds, end=(start+origL)//self.ds)

        return Y

        
    
        
    def _conv(self, x: Tensor, filter_idx = None):     
        origL = x.shape[-1]
        P = self.lenH//2 
        if origL + 2*P > self.sig_len_limit:
            return self._oas(x, filter_idx)
        else:
            return self._direct(x, filter_idx)
    
        
        
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
        conv_dim = len(x.shape) + conv_dim if conv_dim < 0 else conv_dim
        is_last_dim = conv_dim == len(x.shape) - 1
        #swap out dimensions, since self._conv requires the last dimension
        if not is_last_dim:
            x = x.swapaxes(-1, conv_dim)           
        
        y = self._conv(x, filter_idx)
            
        #swap dimension back
        if not is_last_dim:
            y = y.swapaxes(-2, conv_dim)
        return y
    
    


            
            
        
        
    