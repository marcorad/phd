from dataclasses import dataclass, field
from typing import List, Dict, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from sympy.ntheory import factorint

from .morlet import sample_gauss, sample_morlet
from .conv import SeperableConv
from .config import *
from .morlet_sampler import *

from itertools import product
from math import ceil, floor, lcm

from tqdm import tqdm

PI = np.pi

@dataclass
class MorletFilter1D:
    psi: np.array
    lambda_: float
    sigma_w: float
    ds_lambda: float
    ds_phi: float
    fs_in: float
    #fields calculated from input
    f_c: float = field(init=False)
    sigma_f: float = field(init=False)
    bw_w: float = field(init=False)
    bw_f: float = field(init=False)
    fs_out: float = field(init=False)
    
    def __post_init__(self):
        self.f_c = self.lambda_ / 2 / PI
        self.sigma_f = self.sigma_w / 2 / PI
        self.bw_w = self.sigma_w * MORLET_DEFINITION.beta
        self.bw_f = self.sigma_w / 2 / PI
        self.fs_out = self.fs_in / self.ds_lambda
        

class MorletSampler1D:

    def __init__(self, Q, T, fs_in, fstart = None, pol=1.0, include_lpf = False, allow_seperate_ds = True, bw_lim_w = None) -> None:
        self.Q = Q
        self.T = T
        self.fs_in = fs_in
        self.fstart = fstart
        self.include_lpf_in_psi = include_lpf
        self.d_tot = max(int(np.floor(fs_in * T / 2 / MORLET_DEFINITION.beta)), 1) if ENABLE_DS else 1 #total allowed downsampling
        self.polarity = pol
        self.allow_seperate_ds = allow_seperate_ds
        self.is_empty = False
        self.bw_lim_w = bw_lim_w
        self._init_filters()
    
        
    def _init_filters(self):
        #std dev of lpf (phi)
        sigma_phi_w = 2 * PI / self.T
        ws = self.fs_in * 2 * PI   

        if self.fstart:
            lambda_ = self.fstart * 2 * PI
            #ensure fstart does not violate the lowest possible frequency
            assert(lambda_ >= 2 * PI * MORLET_DEFINITION.alpha(self.Q) / self.T)
        else:
            lambda_ = 2 * PI * MORLET_DEFINITION.alpha(self.Q) / self.T
            
        
        bw_lim_w = self.bw_lim_w if self.bw_lim_w else ws/2
        
        print(self.bw_lim_w, ws/2)

        #base std dev at 1 rad/s
        sigma_w = 1 / MORLET_DEFINITION.alpha(self.Q) * (2**(1/self.Q) - 1)

        #std dev at lambda rad/s
        sigma_lambda_w = lambda_ * sigma_w

        #get the time support of the lpf
        nmax = int(np.floor(MORLET_DEFINITION.k * PI * self.fs_in / sigma_phi_w))
        t = np.arange(start=-nmax, stop=nmax+1) / self.fs_in        
        self.t_psi = t
        
        filters: List[MorletFilter1D] = []

        #add the lpf into psi if configured
        if self.include_lpf_in_psi:
            filters += [MorletFilter1D(
                    sample_gauss(t, 1/sigma_phi_w).astype(NUMPY_REAL)/self.fs_in,
                    0,
                    sigma_phi_w,
                    self.d_tot,
                    1,
                    self.fs_in
                )]
            

        #build linear filters for all filters with BWs less than the lpf (phi)
        while sigma_phi_w > sigma_lambda_w and lambda_ < bw_lim_w:
            filters += [MorletFilter1D(
                    sample_morlet(t, lambda_, 1/sigma_phi_w*lambda_, dir = self.polarity).astype(NUMPY_COMPLEX) / self.fs_in,
                    lambda_,
                    sigma_phi_w,
                    self.d_tot,
                    1,
                    self.fs_in
                )]
            lambda_ += MORLET_DEFINITION.alpha(self.Q)*sigma_phi_w
            sigma_lambda_w = lambda_*sigma_w

        #build exponential filters for all filters until the limit is reached
        while lambda_ < bw_lim_w:
            d_lambda, d_phi = self._get_ds_factor(sigma_lambda_w)
            filters += [MorletFilter1D(
                    sample_morlet(t, lambda_, 1/sigma_w, dir = self.polarity).astype(NUMPY_COMPLEX)  / self.fs_in,
                    lambda_,
                    sigma_lambda_w,
                    d_lambda,
                    d_phi,
                    self.fs_in
                )]
            lambda_ *= 2**(1/self.Q)
            sigma_lambda_w = lambda_*sigma_w
            
        if self.include_lpf_in_psi and len(filters) == 1:
            print("Psi only includes the LPF.")
                
        if len(filters) == 0:
            print("Psi is empty - the entire bandwidth has been exhausted.")  
            self.is_empty = True 
                        
        

        #initialise psi
        self.psi = filters      
        
        #change the ds factors if we need a uniform psi output samplerate
        if not self.allow_seperate_ds:
            max_ds_lambda = self.psi[-1].ds_lambda
            max_ds_phi = self.psi[-1].ds_phi
            for f in self.psi:
                f.ds_lambda = max_ds_lambda
                f.ds_phi = max_ds_phi
                
        #get all the lpf sample rates 
        ds_lambdas = set([f.ds_lambda for f in self.psi])
        self.phi = {} #phi is stored as a dictionary, with keys equal to the various DS rates required by the filters
        self.t_phi = {}
        
        #sample lpf for each different psi output sample rate
        for ds in ds_lambdas:    
            fs_psi_out = self.fs_in / ds
            nmax = int(np.floor(MORLET_DEFINITION.k * PI * fs_psi_out / sigma_phi_w))
            t = np.arange(start=-nmax, stop=nmax+1) / fs_psi_out
            phi = sample_gauss(t, 1/sigma_phi_w).astype(NUMPY_REAL) / fs_psi_out                  
            t_phi = t
            self.phi[ds] = phi
            self.t_phi[ds] = t_phi

        #compute filter bandwidths (rad/s)
        self.sigma_phi_w = sigma_phi_w
        self.fs_out = self.fs_in / self.d_tot

        #if polarity is negative
        if self.polarity < 0:
            self.psi.reverse() #reverse the filter order to maintain small-to-large order  
            for f in self.psi:
                f.lambda_ = -f.lambda_ #make the lambdas negative
                f.f_c = -f.f_c #make the freq negative
                
        #get the lest common multiple of all the downsampling factors for uniform padding across filters
        self.psi_ds_max = lcm(*[f.ds_lambda for f in self.psi])
            
    def _get_ds_factor(self, sigma_w):
        bw_f = sigma_w * MORLET_DEFINITION.beta / 2 / PI
        d_lambda = max(int(np.floor(self.fs_in/2/bw_f)),1) if ENABLE_DS else 1
        #find the largest d_lambda such that it divides d_total
        while self.d_tot % d_lambda != 0:
            d_lambda -= 1
        #compute the required phi filter downsampling                
        d_phi = self.d_tot // d_lambda if ENABLE_DS else 1 #downsampling of phi filter        
        return d_lambda, d_phi
        

class MorletSampler1DFull: #for both positive and negative frequency filters, intended to be used in multiple dimensions
    def __init__(self, Q, T, fs_in, include_lpf = True, fstart = None, allow_seperate_ds = True, bw_lim_w = None) -> None:
        self.fb_pos = MorletSampler1D(Q, T, fs_in, pol=+1, include_lpf=include_lpf, fstart = fstart, allow_seperate_ds = allow_seperate_ds, bw_lim_w=bw_lim_w) #include lpf in positive fb
        self.fb_neg = MorletSampler1D(Q, T, fs_in, pol=-1, fstart = fstart, allow_seperate_ds = allow_seperate_ds, bw_lim_w = bw_lim_w)

        #properties from MorletSampler1D for convenience
        self.bw_lim_w = bw_lim_w
        self.Q = Q
        self.T = T
        self.fs_in = fs_in
        self.d_tot = self.fb_neg.d_tot
        self.fs_out = self.fs_in / self.d_tot    
        self.is_empty = self.fb_pos.is_empty and self.fb_neg.is_empty    
        
        #phi and psi filters
        self.psi = self.fb_neg.psi + self.fb_pos.psi        
        self.phi = {}
        for k, v in self.fb_neg.phi.items():
            self.phi[k] = v
        for k, v in self.fb_pos.phi.items():
            self.phi[k] = v
            
        self.psi_ds_max = lcm(self.fb_pos.psi_ds_max, self.fb_neg.psi_ds_max)