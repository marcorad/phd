from enum import IntEnum, auto
from dataclasses import dataclass
from typing import List, Dict, Union
import numpy as np
from torch.nn.common_types import _size_1_t
from . import morlet
from numpy.fft import fft
import matplotlib.pyplot as plt

from torch.nn import Conv1d
from torch import complex128, float64
import torch
import torch.nn as nn

from sympy import factorint

FB_data_type = np.complex128
TORCH_DATA_TYPE = complex128
TORCH_OUTPUT_DATA_TYPE = float64

DEVICE = torch.device('cuda')  
torch.set_default_device(DEVICE)

#If equal to one, then there will be some aliasing, but when equal to 2, there is minimal aliasing
#This parameter controls what is considered the BW with reference to sigma_w, i.e., 1 implies that 1 std. dev. is the bandwidth of the filter
BW_TO_SIGMA_RATIO = 1

class MorletBW(IntEnum):
    _2SIGMA_AT_NEXT_MORLET = auto()
    _1SIGMA_AT_NEXT_MORLET = auto()
    EQUAL_TO_Q = auto()
    EQUAL_TO_2Q = auto()
    
class LambdasScalingMode(IntEnum):
    LINEAR_FOR_LARGER_TIME_SUPPORT = auto()
    ALWAYS_EXPONENTIAL = auto()
    
class DownsampleMode(IntEnum):
    OFF = auto()
    MAXIMUM_UNIFORM = auto()
    OPTIMAL_T = auto()  
    MULTIPLE_OF_LPF = auto()  
  
class Morlet1DConfig:
    '''
    Stores the configuration of a single 1D morlet wavelet.
    '''
    def __init__(self, lambda_: float, f_c: float, psi_time_sigma: float, sigma_t: float, sigma_f: float, sigma_w: float, is_linear: bool, BW_hz: float, downsampling_factor: float, output_fs: float) -> None:        
        self.lambda_: float = lambda_
        self.f_c: float = f_c
        self.psi_time_sigma: float = psi_time_sigma
        self.sigma_t: float = sigma_t
        self.sigma_f: float = sigma_f
        self.sigma_w: float = sigma_w
        self.is_linear: bool = is_linear
        self.BW_hz: float = BW_hz
        self.downsampling_factor: float = downsampling_factor
        self.output_fs: float = output_fs
        
    def __str__(self) -> str:
        return "Morlet @ {:.2f}+-{:.2f} supported on (-{st:.2f}:{st:.2f}) (m = {:d}, lin = {:b})".format(self.f_c, self.BW_hz, self.downsampling_factor, self.is_linear, st=self.sigma_t)
    
def lpf_downsampling(fs, T, oversample):
    return int(np.floor(fs/2/BW_TO_SIGMA_RATIO*T/oversample))

def downsampling_to_T(fs, ds: int, oversample: int) -> float:
    return ds * 2 * BW_TO_SIGMA_RATIO * oversample / fs

def maximum_downsampling(ws, bw_rad, lpf_freq_sigma, oversample):
    final_lpf_downsample = lpf_downsampling(ws, lpf_freq_sigma, oversample)
    # return int(np.floor(min(max(ws/2/bw_rad/oversample, 1), final_lpf_downsample)))
    return int(np.floor(min(max(ws/2/bw_rad, 1), final_lpf_downsample)))

class ScatteringFB1DConfig:  
    '''
    Stores the configuration of a single scattering filter bank.
    '''
    
    def __init__(self, Q, T, fs, scaling_mode = LambdasScalingMode.LINEAR_FOR_LARGER_TIME_SUPPORT, BW_mode = MorletBW._2SIGMA_AT_NEXT_MORLET,
        fstart=0.0, fend=None, approximation_support=5.0, oversampling_factor = 1, downsample_mode=DownsampleMode.MAXIMUM_UNIFORM, T_range: int = None) -> None:
        if fend == None: fend = fs/2
        if T_range == None: T_range = 0.1*T
        self.valid = True
        #choose the BW of each wavelet according to the mode
        psi_w_sigma = 0.0            
        match BW_mode:
            case MorletBW._2SIGMA_AT_NEXT_MORLET: psi_w_sigma = (2**(1 / Q) - 1) / 2
            case MorletBW._1SIGMA_AT_NEXT_MORLET: psi_w_sigma = (2**(1 / Q) - 1)
            case MorletBW.EQUAL_TO_Q: psi_w_sigma = 1 / Q
            case MorletBW.EQUAL_TO_2Q: psi_w_sigma = 2 / Q
        
        psi_t_sigma = 1/psi_w_sigma
        self.ws = 2 * np.pi * fs

        #determine the time supports and starting frequency according to T, fstart, fend
        lambda_0 = 2 * np.pi * fstart
        lambda_end = min(2 * np.pi * fend, fs*2 * np.pi/2)
        
        if downsample_mode == DownsampleMode.OPTIMAL_T:
            #adjust T so that the downsampling factor contains the maximum number of prime factors
            ds_l = lpf_downsampling(fs, (T - T_range), oversampling_factor)
            ds_h = lpf_downsampling(fs, (T + T_range), oversampling_factor)
            best_ds = 1
            most_factors = 0
            for ds in range(ds_l, ds_h+1):
                factors = factorint(ds).items()
                n_factors = 0
                for p in factors: n_factors += p[1]
                if n_factors > most_factors:
                    most_factors = n_factors
                    best_ds = ds
            T = downsampling_to_T(fs, best_ds, oversampling_factor)
                

        self.lpf_w_sigma = 2 * np.pi / T #sets the cut-off frequency to 1/T Hz, with cut-off defined as 1 standard deviation in the frequency domain
        self.lpf_t_sigma = 1 / self.lpf_w_sigma

        if lambda_0 < 2*self.lpf_w_sigma:
            lambda_0 = 2*self.lpf_w_sigma
        

        #start to assemble the centre frequencies
        lambdas: List[float] = []
        morlets: List[Morlet1DConfig] = []

        lambda_ = lambda_0
        if scaling_mode == LambdasScalingMode.LINEAR_FOR_LARGER_TIME_SUPPORT:
            bw = psi_w_sigma * lambda_
            dlambda_ = 2*self.lpf_w_sigma #TODO: set dlambda_ so that similar placement is achieved compared to the exponentially scaled filters
            while bw < self.lpf_w_sigma and lambda_ < lambda_end:
                lambdas += [lambda_]
                psi_time_sigma_lin = self.lpf_t_sigma * lambda_ #ensures that the time support of the wavelet will result in a BW equal to lpf_freq_sigma
                psi_freq_sigma_lin = 1 / psi_time_sigma_lin
                bw_lin = BW_TO_SIGMA_RATIO*psi_freq_sigma_lin*lambda_
                downsample = maximum_downsampling(self.ws, bw_lin, self.lpf_w_sigma, oversampling_factor) 
                morlets +=  [Morlet1DConfig(lambda_, lambda_/2 / np.pi, psi_time_sigma_lin, psi_time_sigma_lin/lambda_, psi_freq_sigma_lin/2 / np.pi, psi_freq_sigma_lin, True, bw_lin/2/np.pi, downsample, fs)]
                lambda_ += dlambda_
                bw = psi_w_sigma * lambda_        
          
        while lambda_ < lambda_end:
            lambdas += [lambda_]   
            bw = psi_w_sigma * lambda_ * BW_TO_SIGMA_RATIO
            downsample = maximum_downsampling(self.ws, bw, self.lpf_w_sigma, oversampling_factor)   
            morlets += [Morlet1DConfig(lambda_, lambda_/2 / np.pi, psi_t_sigma, psi_t_sigma/lambda_, lambda_/psi_t_sigma/2 / np.pi, 1.0/psi_t_sigma, False, bw/2/np.pi, downsample, fs)]
            lambda_ *= 2**(1/Q)        
            
        if len(morlets) == 0: 
            print(f'WARNING: COULD NOT CREATE FILTERBANK AT FS={fs} with T={T} and Q={Q}')
            self.valid = False
            return

        lpf_downsample = lpf_downsampling(fs, T, oversampling_factor)
        print('LPF downsamples by ', lpf_downsample)
        #depending on the downsampling mode, modify the wavelets
        if downsample_mode == DownsampleMode.OFF:
            for m in morlets:
                m.downsampling_factor = 1
            
        elif downsample_mode == DownsampleMode.MAXIMUM_UNIFORM:
            ds = morlets[-1].downsampling_factor
            for m in morlets:
                m.downsampling_factor = ds
                
        elif downsample_mode == DownsampleMode.OPTIMAL_T or downsample_mode == DownsampleMode.MULTIPLE_OF_LPF:        
            ds = morlets[-1].downsampling_factor    
            while lpf_downsample % ds != 0: ds -= 1
            for m in morlets:
                m.downsampling_factor = ds
        
        
        # #calculate the resulting sample frequency after each downsampling step
        # for m in morlets:
        #     m.output_fs = fs/m.downsampling_factor
        #     print(m)        

        ns = int(np.ceil(max(morlets[0].sigma_t*fs*approximation_support, self.lpf_t_sigma*approximation_support*fs)))
                
        self.Q: int = Q
        self.T: float = T
        self.fs: float = fs
        self.fstart: float = fstart
        self.fend: float = fend
        self.lambdas: List[float] = lambdas
        self.BW_mode: MorletBW = BW_mode
        self.scaling_mode: LambdasScalingMode = scaling_mode
        self.morlets: List[Morlet1DConfig] = morlets
        self.number_of_wavelets: int = len(morlets)
        self.oversampling_factor: int = oversampling_factor
        self.max_output_BW_hz: float = morlets[-1].BW_hz
        self.min_time_support_samples: int = ns
        self.approximation_support: float = approximation_support
        self.downsample_mode = downsample_mode
        
        self.filter_downsampling_factor = self.morlets[0].downsampling_factor 
        fs_ds = fs/self.filter_downsampling_factor        
        self.lpf_downsampling_factor = lpf_downsampling(fs_ds, T, self.oversampling_factor)
        self.lpf_output_fs = self.fs / self.lpf_downsampling_factor / self.filter_downsampling_factor
        self.filter_output_fs = fs_ds
   
class ScatteringFB1D:
    '''
    Performs the sampling of scattering filter bank for a specific configuration.
    '''
    def __init__(self, config: ScatteringFB1DConfig, dir = 1.0) -> None:
        self.config = config
        self._sample(dir=dir)
                
    def _sample(self, dir = 1.0):
        supp = self.config.morlets[0].sigma_t * self.config.approximation_support #+- support for IRs in seconds
        N = int(np.ceil(supp*self.config.fs))
        n = np.arange(-N, N+1)
        self.t = n / self.config.fs
        self.filters = np.zeros((2*N + 1, self.config.number_of_wavelets), dtype=FB_data_type)
        for k in range(self.config.number_of_wavelets):
            m = self.config.morlets[k]
            self.filters[:, k] = morlet.sample_morlet(self.t, m.lambda_, m.psi_time_sigma, dir = dir)
            
        self.filter_size = len(n)
        #construct the LPF      
        
        
        fs_d = self.config.filter_output_fs
        supp = self.config.lpf_t_sigma * self.config.approximation_support #+- support for LPF in seconds
        N = int(np.ceil(supp*fs_d))
        n = np.arange(-N, N+1)
        self.t_lpf = n / fs_d
        self.lpf = morlet.sample_gauss(self.t_lpf, self.config.lpf_t_sigma)
        self.lpf_filter_size = len(self.lpf)
        
     
    
            
    def get_fft(self, npoints=10000):
        return fft(self.filters, n=npoints, axis=0)
    
    def plot(self):
        plt.subplot(311)
        plt.plot(np.abs(self.get_fft()))
        plt.subplot(312)
        plt.plot(np.real(self.filters))
        plt.subplot(313)
        plt.plot(np.imag(self.filters))
        plt.show(block=True)
        
      
        
class ScatteringFB1DModule(nn.Module):
    '''
    The torch module that performs the scattering computations for a single 1D filter bank.
    '''
    def __init__(self, fb: ScatteringFB1D, num_filters = None) -> None:
        super().__init__()
        self.fb = fb
        if num_filters == None: num_filters = self.fb.config.number_of_wavelets
        self.num_filters = num_filters
        
        #for convenience        
        self.downsample_mode = fb.config.downsample_mode
        
        self._create_psi()
        self._create_phi()         

            
    def _create_psi(self): 
        self.Psi = Conv1d(
                in_channels=1,
                out_channels=self.num_filters,
                kernel_size=self.fb.filter_size,
                stride=self.fb.config.filter_downsampling_factor,
                padding=self.fb.filter_size//2,
                dtype=TORCH_DATA_TYPE,
                bias=False,    
                device = DEVICE                        
            )
        w = torch.from_numpy(self.fb.filters[:, np.newaxis, 0:self.num_filters].T) #(out_channels, in_channels/groups, L)
        w = w.to(DEVICE)
        self.Psi.weight = nn.Parameter(w, requires_grad = False)         
   
    def _create_phi(self):
        lpf = self.fb.lpf
        size = self.fb.lpf_filter_size        
        self.Phi = Conv1d(
            in_channels=self.num_filters,
            out_channels=self.num_filters,
            kernel_size=size,
            stride=self.fb.config.lpf_downsampling_factor,
            padding=self.fb.lpf_filter_size//2,
            dtype = TORCH_OUTPUT_DATA_TYPE,
            groups = self.num_filters,
            bias = False,    
            device = DEVICE                
        ) 
        w = np.tile(lpf, (self.num_filters, 1, 1))
        w = torch.from_numpy(w)
        w = w.to(DEVICE)
        self.Phi.weight = nn.Parameter(w, requires_grad = False)
    

               

    def U(self, x: torch.Tensor):
        x = x.type(TORCH_DATA_TYPE)
        return torch.abs(self.Psi(x))
    
    def S(self, u: torch.Tensor):
        return self.Phi(u)
    
    def US(self, x: torch.Tensor):
        u = self.U(x)
        s = self.S(u)
        return u, s

class Scattering1D:
    '''
    Configures the filter banks and performs scattering computations for an arbitrary scattering depth.
    '''
    def __init__(self, Q_list, T, fs, oversample=1, fstart = 0.0, fend = None) -> None:
        if not fend: fend = fs
        self.Q_list = Q_list
        self.num_levels = len(Q_list)
        
        self.filter_banks: List[ScatteringFB1D] = []
        #each level has a number of filterbank modules, indexed by the number of filters that should be computed
        self.filter_bank_modules: List[Dict[int, ScatteringFB1DModule]] = [] #stores a FB module that computes the number of filters specified by the key
        self.module_indices: List[List[int]] = [] #stores the number of filters required to compute the coefficients
        
        conf = ScatteringFB1DConfig(Q_list[0], T, fs, approximation_support=3.0, 
                                    oversampling_factor=oversample, fstart=fstart,         
                                    fend=fend, downsample_mode=DownsampleMode.OPTIMAL_T)
        root_fb = ScatteringFB1D(conf)
        self.filter_banks += [root_fb]
        self.module_indices += [[conf.number_of_wavelets]*conf.number_of_wavelets] #all filters in first level use the same module (unused)
        self.T = conf.T #T will adapt to be optimal
        self.fs = fs
        
        
        prev_level = 0
        if self.num_levels > 1:
            for Q in Q_list[1:]:
                print("OUTPUT FS", self.filter_banks[prev_level].config.filter_output_fs)
                conf = ScatteringFB1DConfig(Q, self.T, self.filter_banks[prev_level].config.filter_output_fs, 
                                            approximation_support=3.0, oversampling_factor=oversample, 
                                            downsample_mode=DownsampleMode.MULTIPLE_OF_LPF)
                if not conf.valid: 
                    print(f'ONLY CONFIGURING FOR {prev_level + 1} LEVELS.')
                    self.num_levels = prev_level + 1
                    self.Q_list = self.Q_list[:self.num_levels]
                    break
                self.filter_banks += [ScatteringFB1D(conf)]
                prev_level += 1
                
        #prepare the torch modules
        self.root_module = ScatteringFB1DModule(self.filter_banks[0])
        self.filter_bank_modules += [{self.filter_banks[0].config.number_of_wavelets: self.root_module}]
        for k in range(1, self.num_levels):
            prev_fb = self.filter_banks[k-1]
            curr_fb = self.filter_banks[k]
            
            curr_fb_amounts = []
            modules = {}
            #loop over each filter in the previous FB to determine how many filters the next filterbank must utilize
            for curr_filter in prev_fb.config.morlets:
                bw = curr_filter.BW_hz
                num_filts = 0
                EPS = 1e-6 #TODO: make EPS adaptive to the smallest BW
                for next_filter in curr_fb.config.morlets:
                    #check to see if the filter in the next filterbank will capture significant information
                    if bw < next_filter.f_c - next_filter.BW_hz*BW_TO_SIGMA_RATIO + EPS: break
                    num_filts += 1
                curr_fb_amounts += [num_filts]
                if num_filts not in modules.keys():                    
                    modules[num_filts] = ScatteringFB1DModule(curr_fb, num_filts) if num_filts > 0 else None
                
            self.module_indices += [curr_fb_amounts]
            self.filter_bank_modules += [modules]
            
    def US1(self, x):
        return self.root_module.US(x)
    
    def _transform(self, result, u_prev, path: List[int], curr_fb_index: int):
        if curr_fb_index >= self.num_levels: return
        K = u_prev.shape[1]
        for k in range(K):
            num_filters_required = self.module_indices[curr_fb_index][k]
            module = self.filter_bank_modules[curr_fb_index][num_filters_required]
            if module == None: return
            x = u_prev[:, [k], :]
            u_curr, s_curr = module.US(x)
            p = path + (k, )
            result[p] = (u_curr, s_curr, u_curr.shape[1], u_curr.shape[2])
            self._transform(result, u_curr, p, curr_fb_index+1)
        
    
    def transform(self, x):
        result = {}
        u_curr, s_curr = self.US1(x)
        path = (0,)
        result[path] = (u_curr, s_curr, u_curr.shape[1], u_curr.shape[2])
        self._transform(result, u_curr, path, 1)
        return result
    
    def feature_vector(self, result: Dict):
        n_feats = 0
        for v in result.values():
            n_feats += v[2]
        print(n_feats)
        
                    
                