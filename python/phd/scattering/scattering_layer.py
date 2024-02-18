from dataclasses import dataclass
from typing import List, Dict, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from sympy.ntheory import factorint

from .morlet import sample_gauss, sample_morlet
from .conv import SeperableConv
from .config import *
from .morlet_sampler import *
from .scattering_layer import *

from itertools import product
from math import ceil, floor

from tqdm import tqdm

PI = np.pi

class SeperableScatteringLayer:
    def __init__(self, Q: Union[List[float], float] , T: Union[List[float], float], 
                 fs_in: Union[List[float], float], dims: Union[List[int], int], N: Union[List[int], int],
                 include_on_axis_wavelets = True, fstart: Union[List[float], None] = None,
                 allow_seperate_ds = True, bw_lim_w = None) -> None:
        
        if type(Q) != list: #convert to list if 1D
            Q  = [Q]
            T = [T]
            fs_in = [fs_in]
            dims = [dims]
            N = [N]

        self.Q = Q
        self.T = T
        self.fs_in = fs_in
        self.ndim = len(Q)
        self.include_on_axis_wavelets = include_on_axis_wavelets 
        self.conv_dims = dims   
        self.N = N  
        self.allow_seperate_ds = allow_seperate_ds     
        self.fstart = fstart    
        self.is_empty = False
        self.bw_lim_w = bw_lim_w

        self._init_samplers()
        self._init_conv()
        
    def _init_samplers(self):
        #first dimension is only half of the frequency plane, since we expect input to be real
        self.samplers = [MorletSampler1D(self.Q[0], self.T[0], self.fs_in[0], include_lpf=self.include_on_axis_wavelets, 
                                         fstart= self.fstart[0] if self.fstart else None, allow_seperate_ds=self.allow_seperate_ds,
                                         bw_lim_w=self.bw_lim_w[0] if self.bw_lim_w else None)]
        self.fzero_idx = [0]
        for i in range(1, self.ndim):
            samp = MorletSampler1DFull(self.Q[i], self.T[i], self.fs_in[i], include_lpf=self.include_on_axis_wavelets, 
                                       fstart = self.fstart[i] if self.fstart else None, allow_seperate_ds=self.allow_seperate_ds,
                                       bw_lim_w=self.bw_lim_w[i] if self.bw_lim_w else None)
            self.samplers += [samp]
        
    def _init_conv(self):
        self.conv_psi: Dict[Tuple, SeperableConv] = {}        
        self.conv_phi: Dict[Tuple, SeperableConv] = {}
        filter_idx = []
        self.psi: Dict[Tuple, List[MorletFilter1D]] = {} #for convenience, the filters for all lambda combinations
        self.paths: List[Tuple] = [] #for convenience, the lambda paths across all axes
        for s in self.samplers:
            filter_idx.append([i for i in range(len(s.psi))])
        #generate the cross-product of all the filters
        filter_combinations = list(product(*filter_idx))
        for f_idx in filter_combinations:
            filters: List[MorletFilter1D] = []
            for i, s in enumerate(self.samplers):
                k = f_idx[i]
                f: MorletFilter1D = s.psi[k]
                filters.append(f)
            lambdas = tuple([f.lambda_ for f in filters])
            psi = []
            phi = []
            N_phi = []
            
            #skip the (0,0,...) origin filter
            if all([l == 0 for l in lambdas]): 
                continue 
            #skip on-axis filters with negative components
            if lambdas[0] == 0 and len(lambdas) > 1 and any([l < 0 for l in lambdas[1:]]): 
                continue           
            
            
            #get the requied filter of each axis and its lpf-related data
            for i, f in enumerate(filters):
                psi.append(torch.from_numpy(f.psi))
                phi.append(torch.from_numpy(self.samplers[i].phi[f.ds_lambda]))   
                N_phi.append(ceil(self.N[i] / f.ds_lambda))             
            
            ds = [f.ds_lambda for f in filters]
            ds_phi = [f.ds_phi for f in filters]
            #create a conv object specifically for each filter
            ds_max = [s.psi_ds_max for s in self.samplers]
            self.conv_psi[lambdas] = SeperableConv(psi, self.N, ds, self.conv_dims, ds_max)
            self.conv_phi[lambdas] = SeperableConv(phi, N_phi, ds_phi, self.conv_dims)
            self.psi[lambdas] = filters
        self.paths = list(self.conv_psi.keys())
        print(f'Filterbank has {len(self.paths)} filters.')

    def _conv_psi(self, x: Tensor) -> Dict[Tuple, Tensor]:
        u: Dict[Tuple, Tensor] = {}  
        p0 = self.paths[0]
        conv = self.conv_psi[p0]
        x = conv.fft(x)
        for p in self.paths:
            conv = self.conv_psi[p]
            y = conv.convolve(x, fun_before_ds=torch.abs)           
            u[p] = y
        return u

    def _conv_phi(self, u: Dict[Tuple, Tensor]) -> Dict[Tuple, Tensor]:   
        s: Dict[Tuple, Tensor] = {}     
        for p in self.paths:     
            conv = self.conv_phi[p]
            u_curr = u[p]
            u_curr = conv.fft(u_curr)
            s_curr = conv.convolve(u_curr, fun_before_ds=torch.real)
            s[p] = s_curr
        return s
    
    def US(self, x: Tensor) -> Tuple[Dict[Tuple, Tensor], Dict[Tuple, Tensor]]:
        u = self._conv_psi(x)
        s = self._conv_phi(u)
        return u, s
    

class SeperableScatteringLayerS0:
    def __init__(self, T: List[float], fs_in: List[float], dims: List[int], N: List[int]) -> None:
        self.T = T
        self.fs_in = fs_in
        self.ndim = len(T)
        self.conv_dims = dims
        
        self.phi = []
        self.d_tot = []

        for i in range(self.ndim):
            sigma_phi_w = 2 * PI * MORLET_DEFINITION.beta / self.T[i]
            nmax = int(np.floor(MORLET_DEFINITION.k * PI * self.fs_in[i] / sigma_phi_w))
            t = np.arange(start=-nmax, stop=nmax+1) / self.fs_in[i]
            self.t_psi = t

            phi = sample_gauss(t, 1/sigma_phi_w).astype(NUMPY_REAL)/self.fs_in[i]
            d_tot = max(int(np.floor(fs_in[i] * T[i] / 2 / MORLET_DEFINITION.beta)), 1) if ENABLE_DS else 1 #total allowed downsampling 
            self.d_tot += [d_tot]
            self.phi += [torch.from_numpy(phi)]

        self.conv = SeperableConv(self.phi, N, self.d_tot, self.conv_dims)

        
    def S0(self, x: Tensor):
        x = self.conv.fft(x)
        s0 = self.conv.convolve(x, fun_before_ds=torch.real)      
        return s0
