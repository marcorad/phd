from .config import cfg
from .torch_backend import TorchBackend
from .filterbank import scattering_filterbanks, get_Lambda_set, get_wavelet_filter, filterbank_to_tensor, calculate_padding_1d, get_output_downsample_factor, calculate_sigma_psi_w
from typing import List, Tuple, Dict
from torch import Tensor

class SeparableScattering:
    backend: TorchBackend = TorchBackend()
    def __init__(self, N: List[int], d: List[int], Q: List[List[float]], startfreq: List[float] = None) -> None:
        self.pad = []
        self.Npad = []
        self.Ndim = len(N)
        self.d = d
        self.Q = Q
        self.Nlevels = len(Q)
        assert self.Nlevels <= 3, f'Requested {self.Nlevels} scattering levels. A maximum of 3 levels is supported. Higher levels than 3 are typically not useful.'
        assert all([d[i] <= N[i] for i in range(self.Ndim)]), f'All invariance scales {d} must be <= the signal support {N}.'
        for i in reversed(range(self.Ndim)):
            l, r, n = calculate_padding_1d(N[i], d[i])
            self.pad.extend([l, r])
            self.Npad.append(n)
        self.Npad.reverse()
            
        self.fb = filterbank_to_tensor(scattering_filterbanks(self.Npad, d, Q, startfreq))
        
    def _mul_and_downsample(self, X: Tensor, level: int, input_ds: List[int], lambdas: List[float]):
        mul1d = lambda x, y, dim: self.backend.mul1d(x, y, dim)
        freqds = lambda x, d, dim: self.backend.freq_downsample1d(x, d, dim)
        for dim in range(self.Ndim):
            filter = get_wavelet_filter(self.fb, level, dim, input_ds[dim], lambdas[dim])  
            ds = get_output_downsample_factor(self.fb, level, dim, input_ds[dim], lambdas[dim])
            X = freqds(mul1d(X, filter, dim), ds, dim)
        return X
    
    def _ifft_all(self, X: Tensor):
        dims = [1 + dim for dim in range(self.Ndim)]
        return self.backend.ifft(X, dims)
    
    def _fft_all(self, X: Tensor):
        dims = [1 + dim for dim in range(self.Ndim)]
        return self.backend.fft(X, dims)
    
    def _get_compounded_downsample_factor(self, level, current_ds, lambdas):
        return [get_output_downsample_factor(self.fb, level, dim, current_ds[dim], lambdas[dim]) * current_ds[dim] for dim in range(self.Ndim)]
    
    def _should_prune(self, lambda_filt: List[float], lambda_demod: List[float]):
        for i in range(self.Ndim):
            sigma_psi_w = calculate_sigma_psi_w(self.Q[i])
            beta = cfg.get_beta(self.Q[i])
            # prune only when demodulated filter's bandwith is not at least within beta standard deviations of the current filter
            # note that we use the abs value of the lambdas since lambdas can be positive or negative
            # |----------*----------|
            # |-----ddddd*ddddd-----|
            #            <---->          
            # |----------*--ffxff---|
            #               <->       
            # these intervals must overlap, 
            # * is the centre of the spectrum, d is the significant bandwidth of the demodulated filter (via modulus), and f is the morlet filter under consideration which has a center x
            if abs(lambda_filt) - beta * abs(lambda_filt) * sigma_psi_w > beta * abs(lambda_demod) * sigma_psi_w: return True            
        return False
        
    def _scattering(self, x: Tensor):        
        #function aliases for clarity        
        unpad = lambda x: self.backend.unpad(x)
        pad = lambda x, s: self.backend.pad(x, s)
        fft = lambda x: self._fft_all(x)
        ifft = lambda x: self._ifft_all(x)        
        mulds = lambda x, level, ids, lambdas: self._mul_and_downsample(x, level, ids, lambdas)
        modulus = lambda x: self.backend.modulus(x)        
        
        #pad the tensor
        x = pad(x, self.pad)
        #get the fft of the input signal across all dimensions
        X = fft(x)
        S0 = unpad(ifft(mulds(X, 0, [1 for _ in range(self.Ndim)], [0 for _ in range(self.Ndim)])))
        S = [S0]
        
        l0_compounded_ds = [1 for _ in range(self.Ndim)] #no downsampling on the input
        
        #first level
        Lambda_1 = get_Lambda_set(self.fb, 0, [1]*self.Ndim)
        for lambda1 in Lambda_1:            
            u_1 = modulus(ifft(mulds(X, 0, l0_compounded_ds, lambda1)))
            U_1 = fft(u_1)
            l1_compounded_ds = self._get_compounded_downsample_factor(0, l0_compounded_ds, lambda1)
            s_1 = unpad(ifft(mulds(U_1, 1, l1_compounded_ds, [0 for _ in range(self.Ndim)])).real)
            S.append(s_1)            
            
            if self.Nlevels == 1: continue
            
            #second level
            Lambda_2 = get_Lambda_set(self.fb, 1, l1_compounded_ds)
            for lambda2 in Lambda_2:                
                if self._should_prune(lambda2, lambda1): continue #prune the paths, since downsampling only prunes to an inexact extent
                u_2 = modulus(ifft(mulds(U_1, 1, l1_compounded_ds, lambda2)))
                U_2 = fft(u_2)
                l2_compounded_ds = self._get_compounded_downsample_factor(1, l1_compounded_ds, lambda2)
                s_2 = unpad(ifft(mulds(U_2, 1, l2_compounded_ds, [0 for _ in range(self.Ndim)])).real)
                S.append(s_2)   
                
                if self.Nlevels == 2: continue
                
                #third level
                Lambda_3 = get_Lambda_set(self.fb, 2, l2_compounded_ds)
                for lambda3 in Lambda_3:    
                    if self._should_prune(lambda3, lambda2): continue #prune the paths
                    u_3 = modulus(ifft(mulds(U_2, 2, l2_compounded_ds, lambda3)))
                    U_3 = fft(u_3)
                    l3_compounded_ds = self._get_compounded_downsample_factor(2, l2_compounded_ds, lambda3)
                    s_3 = unpad(ifft(mulds(U_3, 2, l3_compounded_ds, [0 for _ in range(self.Ndim)])).real)
                    S.append(s_3) 
        
        
        
        