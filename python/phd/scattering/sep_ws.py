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

    def __init__(self, Q, T, fs_in, fstart = None, pol=1.0) -> None:
        self.Q = Q
        self.T = T
        self.fs_in = fs_in
        self.fstart = fstart
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
        self.fb_pos = MorletSampler1D(Q, T, fs_in, pol=+1)
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
        self.lambdas = self.fb_pos.lambdas + [0] + [-l for l in self.fb_neg.lambdas]
        self.fc = [l/2/PI for l in self.lambdas]
        self.sigma_phi_w  = self.fb_neg.sigma_phi_w
        self.bw_w = self.fb_pos.bw_w + [self.sigma_phi_w] + self.fb_neg.bw_w
        self.bw_f = [b / 2 / PI for b in self.bw_w]

        #sample phi to fit with psi
        nmax = int(np.floor(MORLET_DEFINITION.k * PI * self.fs_in / self.fb_pos.sigma_phi_w))
        t = np.arange(start=-nmax, stop=nmax+1) / self.fs_in
        phi = sample_gauss(t, 1/self.fb_pos.sigma_phi_w).astype(NUMPY_REAL) /self.fs_in

        self.psi = np.concatenate((self.fb_pos.psi, phi[None, :], self.fb_neg.psi), axis=0, dtype=NUMPY_COMPLEX)
        self.phi = np.array(self.fb_pos.phi, dtype=NUMPY_REAL)


class ScatteringLayer1DS0:
    def __init__(self, T: float, fs_in: float, conv_dim = -1) -> None:
        """Create a convolution layer that computes S0 coefficients: S0 = x * phi. This is the average of x at the invariance scale T.

        Args:
            T (float): Invariance scale
            fs_in (float): Sampling frequency
            conv_dim (int, optional): Dimension to convolve. Defaults to -1.
        """

        assert conv_dim in [0, 2, 3, -1], "Invalid convolution dimension"

        self.d_tot = max(int(np.floor(fs_in * T / 2 / MORLET_DEFINITION.beta)), 1) #total allowed downsampling
        self.fs_in = fs_in
        self.T = T
        self.conv_dim = conv_dim
        sigma_phi_w = 2 * PI * MORLET_DEFINITION.beta / self.T
        nmax = int(np.floor(MORLET_DEFINITION.k * PI * self.fs_in / sigma_phi_w))
        t = np.arange(start=-nmax, stop=nmax+1) / self.fs_in
        self.phi = sample_gauss(t, 1/sigma_phi_w).astype(NUMPY_REAL) /self.fs_in
        self.phi = self.phi[np.newaxis, :] #rehape to (1, L) 
        self.conv = Conv1D(self.phi, conv_dim=conv_dim, ds=self.d_tot) 

    def S0(self, x: Tensor) -> Tensor:
        """Compute S0 coefficients: S0 = x * phi. This is the average of x at the invariance scale T.

        Args:
            x (Tensor): Signal to convolve across the specified dimension (dim in constructor). 
            Must be of shape (Nbatch, 1, Nch, Nx). Any of Nbatch/Nch/Nx dimensions may contain content to convolve.

        Returns:
            Tensor: S0 coefficients.
        """
        return self.conv.conv_multiple(x) 
        

class ScatteringLayer1D:
    
    def __init__(self, Q:float , T: float, fs_in: float, fstart:float=None, full=False, dim = -1) -> None:
        """Construct a scattering convolution layer. Can generate filters that may cover both positive and negative frequencies (full filterbank).
        This layer performs a 1D convolution across a specific dimension, applying the convolution to other dimensions as well.
        Can only handle a maximum of 3 convolution dimensions.

        Args:
            Q (float): Wavelets per octave
            T (float): Invariance scale
            fs_in (float): Input sample frequency
            fstart (float, optional): Start frequency, only applicable to non-full filterbanks. Defaults to None.
            full (bool, optional): Whether to generate both positive and negative frequency filters. Also includes the LPF in the PSI filterbank, intended for seperable convolutions. Defaults to False.
            dim (int, optional): Dimension to convolve across. Input tensors are of shape (Nbatch, 1, Nch, Nx), so valid values are [0, 2, 3]. Defaults to -1 (equivalent to 3).
        """
        assert fstart == None if full else True, "Cannot have an arbitrary starting frequency in a full filter layer"
        assert dim in [0, 2, 3, -1], "Invalid convolution dimension"
        self.sampler: Union[MorletSampler1D, MorletSampler1DFull] = MorletSampler1D(Q, T, fs_in, fstart) if not full else MorletSampler1DFull(Q, T, fs_in)
        self.conv_psi = Conv1D(torch.from_numpy(self.sampler.psi), conv_dim=dim, ds = self.sampler.d_lambda)
        self.conv_phi = Conv1D(torch.from_numpy(self.sampler.phi), conv_dim=dim, ds = self.sampler.d_phi)
        #duplication for convenience
        self.fs_in = fs_in
        self.Q = Q
        self.T = T
        self.fstart = fstart
        self.full = full
        self.conv_dim = dim

    def _conv_psi(self, x, bw_w: float = None):
        filter_idx, _ = self.get_filter_idx(bw_w)
        return self.conv_psi.conv_multiple(x, filter_idx = None if bw_w == None else filter_idx)
    
    def _conv_phi(self, x):
        return self.conv_phi.conv_multiple(x)

    def US(self, x: Tensor, bw_w: float = None) -> Tuple[Tensor, Tensor]:
        """Calculate a single scattering layer's output: U = |x * psi|, S = U * phi. Can only handle a maximum of 3 convolution dimensions.

        Args:
            x (Tensor): Signal to convolve across the specified dimension (dim in constructor). 
            Must be of shape (Nbatch, 1, Nch, Nx). Any of Nbatch/Nch/Nx dimensions may contain content to convolve.
            bw_w (float, optional): Bandwidth of the signal x in the convolution dimension. When None, assumes it occupies full bandwidth. Used for path pruning. Defaults to None (full bandwidth).

        Returns:
            Tuple[Tensor, Tensor]: The layer output: scalogram (U) and scattering coefficients (S)
        """
        filter_idx, _ = self.get_filter_idx(bw_w)
        #compute U
        U = torch.abs(self.conv_psi.conv_multiple(x, filter_idx = None if bw_w == None else filter_idx))
        #compute S from U
        Nfilt = U.shape[1]
        dims = list(U.shape)
        dims[self.conv_dim] = U.shape[self.conv_dim]//self.sampler.d_phi + 1
        S = torch.zeros(dims, dtype=TORCH_REAL)
        for i in range(Nfilt):
            S[:, [i], :, :] = torch.real(self.conv_phi.conv_multiple(U[:, [i], :, :]))

        return U, S   
    

    def get_filter_idx(self, bw_w: float) -> Tuple[List[bool], int]:
        BW = self.fs_in * PI if bw_w == None else bw_w
        #compute which filters should be used by checking to see whether the filter and its bandwidth intersects with the signal bandwidth
        filter_idx = [np.abs(l) - b < BW for l, b in zip(self.sampler.lambdas, self.sampler.bw_w)] #must take abs for negative lambdas
        count = 0
        for f in filter_idx: count += 1 if f else 0
        return filter_idx, count



class SeperableScatteringLayerND:
    def __init__(self, Q: List[float] , T: List[float], fs_in: List[float], dims: List[int]) -> None:
        self.Q = Q
        self.T = T
        self.fs_in = fs_in
        self.ndim = len(Q)
        self.conv_dims = dims

        assert self.ndim <= 3, "Only up to 3D is supported"
        assert self.ndim == len(T) and self.ndim == len(fs_in) and self.ndim == len(dims), "Q, T, fs_in and dims must all be the same size"
        dimcheck = True
        for d in dims: dimcheck = dimcheck and d in [0, 2, 3, -1]
        assert dimcheck, "Only dimensions [0, 2, 3, -1] are allowed for convolutions"

        self.conv_layers: List[ScatteringLayer1D] = []
        for i in range(self.ndim):
            #we only expect real signals, meaning we must cover only half frequency plane
            #by default, the first specified dimension only gets half coverage
            self.conv_layers += [ScatteringLayer1D(Q[i], T[i], fs_in[i], dim=self.conv_dims[i], full=(i!=0))]

    def _conv_dim_psi(self, C: Tensor, layer_idx: int, bw_w: List[float]):
        nfilt = C.shape[1]
        res: List[Tensor] = []
        for i in range(nfilt):
            res += [self.conv_layers[layer_idx]._conv_psi(C[:, [i], :, :], bw_w[layer_idx])]

        return torch.concat(res, dim=1)
    
    def _conv_dim_phi(self, C: Tensor, layer_idx: int):
        nfilt = C.shape[1]
        res: List[Tensor] = []
        for i in range(nfilt):
            res += [self.conv_layers[layer_idx]._conv_phi(C[:, [i], :, :])]

        return torch.concat(res, dim=1)
    


    def US(self, x: Tensor, bw_w: List[float] = None):
        if bw_w == None: bw_w = [None for _ in range(self.ndim)]
        assert len(bw_w) == self.ndim, "Bandwiths must be provided for all axes"


        #Compute U = abs((x * psi1) * psi2 * ...)        
        c = self._conv_dim_psi(x, 0, bw_w)
        for layer_idx in range(1, self.ndim):
            c = self._conv_dim_psi(c, layer_idx, bw_w)
        U = torch.abs(c)

        #Compute S = U * phi
        S = self._conv_dim_phi(U, 0)
        for layer_idx in range(1, self.ndim):
            S = self._conv_dim_phi(torch.real(S), layer_idx)

        return U, torch.real(S)  


class SeperableScatteringLayerNDS0:
    def __init__(self, T: List[float], fs_in: List[float], dims: List[int]) -> None:
        self.T = T
        self.fs_in = fs_in
        self.ndim = len(T)
        self.conv_dims = dims

        assert self.ndim <= 3, "Only up to 3D is supported"
        assert self.ndim == len(fs_in) and self.ndim == len(dims), "Q, T, fs_in and dims must all be the same size"
        dimcheck = True
        for d in dims: dimcheck = dimcheck and d in [0, 2, 3, -1]
        assert dimcheck, "Only dimensions [0, 2, 3, -1] are allowed for convolutions"

        self.conv_layers: List[ScatteringLayer1DS0] = [ScatteringLayer1DS0(T[i], fs_in[i], dims[i]) for i in range(self.ndim)]

    def S0(self, x):
        S0 = self.conv_layers[0].S0(x)
        for i in range(self.ndim):
            S0 = self.conv_layers[i].S0(S0)
        return S0
        

class SeperableWS:
    
    def __init__(self, Q, T, fs) -> None:
        self.J = len(Q)
        assert(self.J == len(T) and self.J == len(fs))