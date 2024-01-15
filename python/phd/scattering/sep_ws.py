# seperable wavelet scattering

from dataclasses import dataclass
from typing import List, Dict, Union

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

    def __init__(self, Q, T, fs_in, fstart = None, pol=1.0) -> None:
        self.Q = Q
        self.T = T
        self.fs_in = fs_in
        self.fstart = fstart
        self.d_tot = max(int(np.floor(fs_in * T / 2 / MORLET_DEFINITION.beta)), 1) #total allowed downsampling
        self.polarity = pol
        self._init_filters()
        self.is_conv = False
    
        
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

        #build linear filters for all filters with BWs less than the lpf (phi)
        while sigma_phi_w > sigma_lambda_w and lambda_ < ws/2:
            lambdas += [lambda_]
            psi += [sample_morlet(t, lambda_, 1/sigma_phi_w*lambda_, self.polarity) / self.fs_in]
            lambda_ += MORLET_DEFINITION.alpha(self.Q)*sigma_phi_w
            sigma_lambda_w = lambda_*sigma_w

        #build exponential filters for all filters until the limit is reached
        while lambda_ < ws/2:
            lambdas += [lambda_]
            psi += [sample_morlet(t, lambda_, 1/sigma_w, self.polarity) / self.fs_in]
            lambda_ *= 2**(1/self.Q)
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
        self.phi = sample_gauss(t, 1/sigma_phi_w)/self.fs_psi_out
        self.phi = self.phi[None, :] #rehape to (1, L)
        self.t_phi = t

    def _init_downsampling_factors(self):
        max_bw = self.max_sigma_lambda_w * MORLET_DEFINITION.beta
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
        self.fb_pos = MorletSampler1D(Q, T, fs_in, pol=+1)
        self.fb_neg = MorletSampler1D(Q, T, fs_in, pol=-1)
        self.psi = np.concatenate((self.fb_pos.psi, self.fb_pos.phi, self.fb_neg.psi), axis=0)
        self.phi = self.fb_pos.phi
        #TODO: add other properties from MorletSampler1D


        

class SeperableWS:
    
    def __init__(self, Q, T, fs) -> None:
        self.Ndim = len(Q)
        assert(self.Ndim == len(T) and self.Ndim == len(fs))