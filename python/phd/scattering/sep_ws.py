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

from itertools import product

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
        self.include_lpf_in_psi = include_lpf
        self.d_tot = max(int(np.floor(fs_in * T / 2 / MORLET_DEFINITION.beta)), 1) #total allowed downsampling
        self.polarity = pol
        self._init_filters()
    
        
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
        d_lambda = max(int(np.floor(self.fs_in/2/max_bw)),1) if ENABLE_DS else 1

        #find the largest d_lambda such that it divides d_total
        while self.d_tot % d_lambda != 0:
            d_lambda -= 1
        
        # the total downsampling d_tot = d_lambda * d_phi
        # S(x, lambda) = downsample(   downsample(|x * psi|, d_lambda)   *   phi,   d_phi)
        self.d_lambda = d_lambda #downsampling of psi filter
        self.d_phi = self.d_tot // d_lambda if ENABLE_DS else 1 #downsampling of phi filter
        self.fs_psi_out = self.fs_in / self.d_lambda #output frequency of the psi filters
        self.fs_out = self.fs_in / self.d_tot #output frequency of the phi filters

class MorletSampler1DFull: #for both positive and negative frequency filters, intended to be used in multiple dimensions
    def __init__(self, Q, T, fs_in, include_lpf = True) -> None:
        self.fb_pos = MorletSampler1D(Q, T, fs_in, pol=+1, include_lpf=include_lpf) #include lpf in positive fb
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
        self.bw_w = self.fb_pos.bw_w + self.fb_neg.bw_w
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
                 fs_in: Union[List[float], float], dims: Union[List[int], int],
                 include_on_axis_wavelets = True) -> None:
        
        if type(Q) != list: #convert to list if 1D
            Q  = [Q]
            T = [T]
            fs_in = [fs_in]
            dims = [dims]

        self.Q = Q
        self.T = T
        self.fs_in = fs_in
        self.ndim = len(Q)
        self.include_on_axis_wavelets = include_on_axis_wavelets

        #convert conv dims to concrete dimension values        
        self.conv_dims =  [(self.ndim + d if d < 0 else d) for d in dims]   

        #first dimension is only half of the frequency plane, since we expect input to be real
        #NOTE: when convolving, take care to exclude the path where the filter is at 0 (in d dimensions)
        self.samplers = [MorletSampler1D(Q[0], T[0], fs_in[0], include_lpf=include_on_axis_wavelets)]
        self.fzero_idx = [0]
        for i in range(1, self.ndim):
            samp = MorletSampler1DFull(Q[i], T[i], fs_in[i], include_lpf=include_on_axis_wavelets)
            self.samplers += [samp]
            self.fzero_idx += [len(samp.lambdas)//2]

        self.conv_psi = [Conv1D(torch.from_numpy(s.psi), s.d_lambda) for s in self.samplers]
        self.conv_phi = [Conv1D(torch.from_numpy(s.phi), s.d_phi) for s in self.samplers]

        self._compute_lambda_pairs()

    def _compute_lambda_pairs(self):
        L = [s.lambdas for s in self.samplers]        
        self.filter_lambda_pairs = [list(p) for p in list(product(*L))]

        L = [s.bw_w for s in self.samplers]        
        self.filter_bw_pairs = [list(p) for p in list(product(*L))]

        # select_idx = []

        # for i in range(len(self._raw_filter_lambda_pairs)):
        #     p = self._raw_filter_lambda_pairs[i]
        #     if p[0] == 0:                                
        #         for d in p[1:]:
        #             if d > 0:
        #                 select_idx += [i]
        #                 break                    
        #     else:
        #         select_idx += [i]

        # idx = np.array(select_idx, dtype=np.int32)
        # self.filter_lambda_pairs = []
        # for i in range(len(self._raw_filter_lambda_pairs)):
        #     if i in idx: self.filter_lambda_pairs += [self._raw_filter_lambda_pairs[i]]

        # # print(self.filter_lambda_pairs)
        # self.select_idx = torch.from_numpy(idx).cuda()

    
    def _select_filters(self, bw_w):

        if bw_w == None: 
            bw_w = [fs*PI for fs in self.fs_in]                    

        #decide which filters to keep according to bandwidth
        filter_idx = []
        filter_lambdas = []
        filter_bws = []
        for i in range(self.ndim):
            lambdas = np.array(self.samplers[i].lambdas)
            bws = np.array(self.samplers[i].bw_w)
            idx = np.argwhere(bw_w[i] * MORLET_DEFINITION.beta > np.abs(lambdas) - bws * (MORLET_DEFINITION.beta + 1e-6)).astype(np.int32)
            # idx = np.argwhere(bw_w[i] * MORLET_DEFINITION.beta > np.abs(lambdas)).astype(np.int32)
            idx = idx.flatten()
            filter_idx += [idx]
            filter_lambdas += [lambdas.flatten()[idx].tolist()]
            filter_bws += [bws.flatten()[idx].tolist()]

        #if the multiple convs are flattened, discard the filters which are unnecessary
        #this only yields results when include_on_axis_wavelets is true 
        cart_product_keep_indices = []
        flat_lambdas = list(product(*filter_lambdas))
        flat_bws = list(product(*filter_lambdas))
        for i in range(len(flat_bws)):
            p = flat_lambdas[i]
            if p[0] == 0:                                
                for d in p[1:]:
                    if d > 0:
                        cart_product_keep_indices += [i]
                        break                    
            else:
                cart_product_keep_indices += [i]

        return filter_idx, np.array(cart_product_keep_indices).flatten(), flat_bws
    

    def _conv_psi(self, x: Tensor, bw_w = None):

        #TODO: do bandwidth selection
        
        filter_bw_keep, cart_product_keep_indices, flat_bws = self._select_filters(bw_w)
        
        if flat_bws == [] or cart_product_keep_indices.shape[0] == 0: return None, None

        #cross product of seperable filters
        orig_shape = list(x.shape)
        for c, d, idx in zip(self.conv_psi, self.conv_dims, filter_bw_keep): 
            x = c.convolve(x, conv_dim=d, filter_idx=torch.from_numpy(idx))            
            
            
        #x is now of shape (..., Nf_1, ..., Nf_ndim)
        #add all the filters into 1 dimension so that x is of shape (..., Nf_1*...*Nf_ndim)
        shape = list(x.shape)[:len(orig_shape)]
        p = 1
        filter_shape = list(x.shape)[len(orig_shape):]
        for d in filter_shape: p *= d
        shape += [p]    
        x = x.reshape(shape)

        #the index of the 0 filter is always at 0
        return torch.index_select(x, dim=-1, index=torch.from_numpy(cart_product_keep_indices).cuda()), flat_bws

    def _conv_phi(self, x: Tensor):
        for c, d in zip(self.conv_phi, self.conv_dims):
            #there is only one filter, remove the 1 at the end
            #only keep the real parts here (due to fft accuracy)
            x = c.convolve(x, conv_dim=d).squeeze(dim=-1).real 
        return x
    
    def US(self, x: Tensor, bw_w = None, nonlin=torch.abs):
        U, S, _ = self._US_info(x, bw_w)
        return U, S
    
    def _US_info(self, x: Tensor, bw_w = None, nonlin=torch.abs):
        U, flat_bws = self._conv_psi(x, bw_w)
        if U == None: return None, None, None
        U = nonlin(U) if nonlin else U
        S = self._conv_phi(U)
        return U, S, flat_bws
    
    def _get_filter_idx(self, bw_w: float) -> Tuple[List[bool], int]:
        BW = self.fs_in * PI if bw_w == None else bw_w
        #compute which filters should be used by checking to see whether the filter and its bandwidth intersects with the signal bandwidth
        filter_idx = [np.abs(l) - b < BW for l, b in zip(self.sampler.lambdas, self.sampler.bw_w)] #must take abs for negative lambdas
        count = 0
        for f in filter_idx: count += 1 if f else 0
        return filter_idx, count
    
    def get_psi_output_fs(self):
        fs = []
        for s in self.samplers:
            fs += [s.fs_psi_out]
        return fs

class SeperableScatteringLayerS0:
    def __init__(self, T: List[float], fs_in: List[float], dims: List[int]) -> None:
        self.T = T
        self.fs_in = fs_in
        self.ndim = len(T)
        self.conv_dims = dims

        self.conv: List[Conv1D] = []
        self.phi = []
        self.d_tot = []

        for i in range(self.ndim):
            sigma_phi_w = 2 * PI * MORLET_DEFINITION.beta / self.T[i]
            nmax = int(np.floor(MORLET_DEFINITION.k * PI * self.fs_in[i] / sigma_phi_w))
            t = np.arange(start=-nmax, stop=nmax+1) / self.fs_in[i]
            self.t_psi = t

            phi = sample_gauss(t, 1/sigma_phi_w).astype(NUMPY_REAL)/self.fs_in[i]
            phi = phi[np.newaxis, :] #rehape to (1, L) 
            d_tot = max(int(np.floor(fs_in[i] * T[i] / 2 / MORLET_DEFINITION.beta)), 1) if ENABLE_DS else 1 #total allowed downsampling

            conv = Conv1D(torch.from_numpy(phi), ds=d_tot) 

            self.conv += [conv]
            self.d_tot += [d_tot]
            self.phi += [phi]

        
    def S0(self, x: Tensor):
        S0 = self.conv[0].convolve(x, conv_dim=self.conv_dims[0])
        for n in range(1, self.ndim):
            #will add another empty dimension, so remove it
            #we want the shape to be (...xd, 1)
            S0 = self.conv[n].convolve(S0, conv_dim=self.conv_dims[n]).squeeze(-1)
        return S0.real
        

class SeperableWaveletScattering:
    
    def __init__(self, Q, T, fs, dims, include_on_axis_wavelets=True, prune=True) -> None:
        self.J = len(Q)
        self.dims = dims
        self.Q = Q
        self.T = T
        self.fs = fs
        self.prune = prune
        self.s0_layer = SeperableScatteringLayerS0(T, fs, dims)
        
        self.sws_layers: List[SeperableScatteringLayer] = [SeperableScatteringLayer(Q[0], T, fs, dims, include_on_axis_wavelets)]
        for j in range(1, self.J):
            self.sws_layers += [SeperableScatteringLayer(
                Q[j], T, self.sws_layers[j-1].get_psi_output_fs(), dims, include_on_axis_wavelets
                )]
            
    def _US_no_prune(self, x: Tensor):
        s0 = self.s0_layer.S0(x)
        u1, s1 = self.sws_layers[0].US(x)
        S = [s0, s1]
        U = [u1]
        for j in range(1, self.J):
            u, s = self.sws_layers[j].US(U[j-1])
            
            u = torch.flatten(u, start_dim=-2)
            s = torch.flatten(s, start_dim=-2)
            U += [u]
            S += [s]
        return U, torch.concat(S, dim=-1)
    

    def _US_prune_rec(self, U, S, curr_bw, curr_level: int, u_curr: Tensor, discard_U):
        if curr_level == self.J: return        
        n_filt = u_curr.shape[-1]
        U_temp = []
        for n in range(n_filt):
            bw = curr_bw[n]
            u_n = u_curr[...,n]
            u, s, this_bw = self.sws_layers[curr_level]._US_info(u_n, bw)
            if u != None:
                S += [s]
                self._US_prune_rec(U, S, this_bw, curr_level+1, u, discard_U)
                if not discard_U: U_temp += [u]
                else: del u
        if not discard_U: U += U_temp

    
    def _US_prune(self, x: Tensor, discard_U = True) -> Union[Tuple[List[Tensor], Tensor], Tensor]:
        s0 = self.s0_layer.S0(x)
        u1, s1, curr_bws = self.sws_layers[0]._US_info(x)
        U = None if discard_U else [u1]
        S = [s0, s1]
        self._US_prune_rec(U, S, curr_bws, 1, u1, discard_U)   
        if discard_U:
            del u1 
            return torch.concat(S, dim=-1)
        else:
            return U, torch.concat(S, dim=-1)
    
    def scatteringTransform(self, x: Tensor, batch_size = None, batch_dim = None, discard_U = True, prune = True, flatten = False) -> Tuple[Union[None, List[Tensor]], Tensor]:
        
        if batch_dim == None:
            if prune:
                return self._US_prune(x, discard_U)
            else:
                return self._US_no_prune()
        else:
            n_samples = x.shape[batch_dim]
            S_batch = []
            i = 0
            while i < n_samples:
                #only supports pruning and U discard
                xb = torch.slice_copy(x, batch_dim, start=i, end=min(i+batch_size, n_samples))
                s = self._US_prune(xb, True).cpu()  
                            
                S_batch += [s]
                i += batch_size
                print(i)
            return torch.concat(S_batch, dim=batch_dim)
                

            
        