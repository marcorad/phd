# seperable wavelet scattering

from dataclasses import dataclass
from typing import List, Dict, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from sympy.ntheory import factorint

from .morlet import sample_gauss, sample_morlet
from .conv import Conv1D
from .config import *

PI = np.pi

def optimise_T(T_nom, fs, eps = 0.1):
    beta = MORLET_DEFINITION.beta
    d1 = int(np.floor(fs * T_nom * (1-eps) / 2 / beta))
    d2 = int(np.floor(fs * T_nom * (1+eps) / 2 / beta))
    kmax = 1
    d_hat = d1

    for d in range(d1, d2 + 1):
        M = list(factorint(d).values())
        k = np.sum(M)
        if k > kmax:
            kmax = k
            d_hat = d

    T_opt = 2 * beta * d_hat / fs
    return T_opt


class MorletSampler1D:

    def __init__(self, Q, T, fs_in, fstart = None, pol=1.0, include_lpf = False) -> None:
        self.Q = Q
        self.T = T
        self.fs_in = fs_in
        self.fstart = fstart
        self.d_tot = max(int(np.floor(fs_in * T / 2 / MORLET_DEFINITION.beta)), 1) #total allowed downsampling
        self.polarity = pol
        self._init_filters()
        self.include_lpf_in_psi = include_lpf
    
        
    def _init_filters(self):
        #std dev of lpf (phi)
        sigma_phi_w = 2 * PI * MORLET_DEFINITION.beta / self.T
        ws = self.fs_in * 2 * PI

        if self.fstart:
            lambda_ = self.fstart * 2 * PI
            #ensure fstart does not violate the lowest possible frequency
            assert(lambda_ >= 2 * PI * MORLET_DEFINITION.alpha(self.Q) / self.T)
        else:
            lambda_ = 2 * PI * MORLET_DEFINITION.alpha(self.Q) / self.T
        
        lambdas = []
        psi = []

        #base std dev at 1 rad/s
        sigma_w = 1 / MORLET_DEFINITION.alpha(self.Q) * (2**(1/self.Q) - 1)

        #std dev at lambda rad/s
        sigma_lambda_w = lambda_ * sigma_w

        #get the time support of the lpf
        nmax = int(np.floor(MORLET_DEFINITION.k * PI * self.fs_in / sigma_phi_w))
        t = np.arange(start=-nmax, stop=nmax+1) / self.fs_in
        self.t_psi = t

        sigmas = [] #freq sigmas

        #add the lpf into psi if configured
        if self.include_lpf_in_psi:
            psi += [sample_gauss(t, 1/sigma_phi_w).astype(NUMPY_REAL)/self.fs_in]
            lambdas += [0]
            sigmas += [sigma_phi_w]

        #build linear filters for all filters with BWs less than the lpf (phi)
        while sigma_phi_w > sigma_lambda_w and lambda_ < ws/2:
            lambdas += [lambda_]
            psi += [sample_morlet(t, lambda_, 1/sigma_phi_w*lambda_, dir = self.polarity).astype(NUMPY_COMPLEX) / self.fs_in]
            lambda_ += MORLET_DEFINITION.alpha(self.Q)*sigma_phi_w
            sigmas += [sigma_phi_w]
            sigma_lambda_w = lambda_*sigma_w

        #build exponential filters for all filters until the limit is reached
        while lambda_ < ws/2:
            lambdas += [lambda_]
            psi += [sample_morlet(t, lambda_, 1/sigma_w, dir = self.polarity).astype(NUMPY_COMPLEX)  / self.fs_in]
            lambda_ *= 2**(1/self.Q)
            sigmas += [sigma_lambda_w]
            sigma_lambda_w = lambda_*sigma_w

        #initialise psi
        self.psi = np.array(psi) #(Nfilt, L)
        self.lambdas = lambdas
        self.fc = (np.array(lambdas)/2/PI).tolist()
        self.max_sigma_lambda_w = lambdas[-1] * sigma_w
        self._init_downsampling_factors() #dowsampling resulting from psi configuration

        #sample lpf
        nmax = int(np.floor(MORLET_DEFINITION.k * PI * self.fs_psi_out / sigma_phi_w))
        t = np.arange(start=-nmax, stop=nmax+1) / self.fs_psi_out
        self.phi = sample_gauss(t, 1/sigma_phi_w).astype(NUMPY_REAL) /self.fs_psi_out
        self.phi = self.phi[np.newaxis, :] #rehape to (1, L)        
        self.t_phi = t

        #compute filter bandwidths (rad/s)
        self.bw_w = [MORLET_DEFINITION.beta * s for s in sigmas]
        self.bw_f = [b / 2 / PI for b in self.bw_w]
        self.sigma_phi_w = sigma_phi_w

    def _init_downsampling_factors(self):
        max_bw = self.max_sigma_lambda_w * MORLET_DEFINITION.beta / 2 / PI
        d_lambda = max(int(np.floor(self.fs_in/2/max_bw)),1)

        #find the largest d_lambda such that it divides d_total
        while self.d_tot % d_lambda != 0:
            d_lambda -= 1
        
        # the total downsampling d_tot = d_lambda * d_phi
        # S(x, lambda) = downsample(   downsample(|x * psi|, d_lambda)   *   phi,   d_phi)
        self.d_lambda = d_lambda #downsampling of psi filter
        self.d_phi = self.d_tot // d_lambda #downsampling of phi filter
        self.fs_psi_out = self.fs_in / self.d_lambda #output frequency of the psi filters
        self.fs_out = self.fs_in / self.d_tot #output frequency of the phi filters

class MorletSampler1DFull: #for both positive and negative frequency filters, intended to be used in multiple dimensions
    def __init__(self, Q, T, fs_in) -> None:
        self.fb_pos = MorletSampler1D(Q, T, fs_in, pol=+1, include_lpf=True) #include lpf in positive fb
        self.fb_neg = MorletSampler1D(Q, T, fs_in, pol=-1)

        #properties from MorletSampler1D
        self.Q = Q
        self.T = T
        self.fs_in = fs_in
        self.d_tot = self.fb_neg.d_tot
        self.d_lambda = self.fb_neg.d_lambda
        self.d_phi = self.fb_pos.d_phi
        self.fs_psi_out = self.fb_neg.fs_psi_out
        self.fs_out = self.fb_neg.fs_out
        self.lambdas = self.fb_pos.lambdas + [-l for l in self.fb_neg.lambdas]
        self.fc = [l/2/PI for l in self.lambdas]
        self.sigma_phi_w  = self.fb_neg.sigma_phi_w
        self.bw_w = self.fb_pos.bw_w + [self.sigma_phi_w] + self.fb_neg.bw_w
        self.bw_f = [b / 2 / PI for b in self.bw_w]  

        self.psi = np.concatenate((self.fb_pos.psi, self.fb_neg.psi), axis=0, dtype=NUMPY_COMPLEX)
        self.phi = np.array(self.fb_pos.phi, dtype=NUMPY_REAL)
    

#https://stackoverflow.com/questions/4114461/does-numpy-have-an-inverse-of-unravel-index
def ravel_index(pos, shape):
    res = 0
    acc = 1
    for pi, si in zip(reversed(pos), reversed(shape)):
        res += pi * acc
        acc *= si
    return res

class SeperableScatteringLayer:
    def __init__(self, Q: Union[List[float], float] , T: Union[List[float], float], 
                 fs_in: Union[List[float], float], dims: Union[List[int], int]) -> None:
        
        if type(Q) != list: #convert to list if 1D
            Q  = [Q]
            T = [T]
            fs_in = [fs_in]
            dims = [dims]

        self.Q = Q
        self.T = T
        self.fs_in = fs_in
        self.ndim = len(Q)

        #convert conv dims to concrete dimension values        
        self.conv_dims =  [(self.ndim - d if d < 0 else d) for d in dims]   

        #first dimension is only half of the frequency plane, since we expect input to be real
        #NOTE: when convolving, take care to exclude the path where the filter is at 0 (in d dimensions)
        self.samplers = [MorletSampler1D(Q[0], T[0], fs_in[0], include_lpf=True)]
        self.fzero_idx = [0]
        for i in range(1, self.ndim):
            samp = MorletSampler1DFull(Q[i], T[i], fs_in=[i])
            self.samplers += [samp]
            self.fzero_idx += [len(samp.lambdas)//2]

        self.conv_psi = [Conv1D(s.psi, s.d_lambda) for s in self.samplers]
        self.conv_phi = [Conv1D(s.phi, s.d_phi) for s in self.samplers]

    def _conv_psi(self, x: Tensor, bw_w = None):
        #cross product of seperable filters
        orig_shape = list(x.shape)
        for c, d in zip(self.conv_psi, self.conv_dims):
            x = c.convolve(x, conv_dim=d)
        #x is now of shape (..., Nf_1, ..., Nf_ndim)
        #add all the filters into 1 dimension so that x is of shape (..., Nf_1*...*Nf_ndim)
        shape = list(x.shape)[:len(orig_shape)]
        p = 1
        filter_shape = list(x.shape)[len(orig_shape):]
        for d in filter_shape: p *= d
        shape += [p]    
        x = x.reshape(shape)

        #compute the index of the 0 filter
        idx = ravel_index(self.fzero_idx, filter_shape)
        #remove the filter at 0
        return torch.cat((torch.slice_copy(x, dim=-1, start=0, end=idx), torch.slice_copy(x, dim=-1, start=idx+1)), dim=-1)

    def _conv_phi(self, x: Tensor):
        for c, d in zip(self.conv_phi, self.conv_dims):
            #there is only one filter, remove the 1 at the end
            #only keep the real parts here (due to fft accuracy)
            x = c.convolve(x, conv_dim=d).squeeze(dim=-1).real 
        return x
    
    def US(self, x: Tensor, bw_w = None):
        U = torch.abs(self._conv_psi(x, bw_w))
        S = self._conv_phi(U)
        return U, S
    
    def _get_filter_idx(self, bw_w: float) -> Tuple[List[bool], int]:
        BW = self.fs_in * PI if bw_w == None else bw_w
        #compute which filters should be used by checking to see whether the filter and its bandwidth intersects with the signal bandwidth
        filter_idx = [np.abs(l) - b < BW for l, b in zip(self.sampler.lambdas, self.sampler.bw_w)] #must take abs for negative lambdas
        count = 0
        for f in filter_idx: count += 1 if f else 0
        return filter_idx, count

# class SeperableScatteringLayerNDS0:
#     def __init__(self, T: List[float], fs_in: List[float], dims: List[int]) -> None:
#         self.T = T
#         self.fs_in = fs_in
#         self.ndim = len(T)
#         self.conv_dims = dims

#         assert self.ndim <= 3, "Only up to 3D is supported"
#         assert self.ndim == len(fs_in) and self.ndim == len(dims), "Q, T, fs_in and dims must all be the same size"
#         dimcheck = True
#         for d in dims: dimcheck = dimcheck and d in [0, 2, 3, -1]
#         assert dimcheck, "Only dimensions [0, 2, 3, -1] are allowed for convolutions"

#         self.conv_layers: List[ScatteringLayer1DS0] = [ScatteringLayer1DS0(T[i], fs_in[i], dims[i]) for i in range(self.ndim)]

#     def S0(self, x):
#         S0 = self.conv_layers[0].S0(x)
#         for i in range(self.ndim):
#             S0 = self.conv_layers[i].S0(S0)
#         return S0
        

class SeperableWS:
    
    def __init__(self, Q, T, fs) -> None:
        self.J = len(Q)
        assert(self.J == len(T) and self.J == len(fs))