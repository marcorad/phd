from enum import IntEnum, auto
from dataclasses import dataclass
from typing import List, Dict, Union
import numpy as np
from torch.nn.common_types import _size_1_t
from scattering import morlet
from numpy.fft import fft
import matplotlib.pyplot as plt

from torch.nn import Conv1d
from torch import complex128, float64
import torch
import torch.nn as nn

FB_data_type = np.complex128
torch_data_type = complex128
torch_output_data_type = float64

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
    MAXIMUM_SPLIT = auto()
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
    
def lpf_downsampling(ws, lpf_freq_sigma, oversample):
    return int(np.floor(ws/4/lpf_freq_sigma/oversample))

def maximum_downsampling(ws, bw_rad, lpf_freq_sigma, oversample):
    final_lpf_downsample = lpf_downsampling(ws, lpf_freq_sigma, oversample)
    # return int(np.floor(min(max(ws/2/bw_rad/oversample, 1), final_lpf_downsample)))
    return int(np.floor(min(max(ws/2/bw_rad, 1), final_lpf_downsample)))

class ScatteringFB1DConfig:  
    '''
    Stores the configuration of a single scattering filter bank.
    '''
    
    def __init__(self, Q, T, fs, scaling_mode = LambdasScalingMode.LINEAR_FOR_LARGER_TIME_SUPPORT, BW_mode = MorletBW._2SIGMA_AT_NEXT_MORLET,
        fstart=0.0, fend=None, approximation_support=5.0, oversampling_factor = 1, downsample_mode=DownsampleMode.MAXIMUM_UNIFORM) -> None:
        if fend == None: fend = fs
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
        lambda_end = min(2 * np.pi * fend, fs*2 * np.pi)/2

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
            while bw < self.lpf_w_sigma:
                lambdas += [lambda_]
                psi_time_sigma_lin = self.lpf_t_sigma * lambda_ #ensures that the time support of the wavelet will result in a BW equal to lpf_freq_sigma
                psi_freq_sigma_lin = 1 / psi_time_sigma_lin
                bw_lin = 2*psi_freq_sigma_lin*lambda_
                downsample = maximum_downsampling(self.ws, bw_lin, self.lpf_w_sigma, oversampling_factor) 
                morlets +=  [Morlet1DConfig(lambda_, lambda_/2 / np.pi, psi_time_sigma_lin, psi_time_sigma_lin/lambda_, psi_freq_sigma_lin/2 / np.pi, psi_freq_sigma_lin, True, bw_lin/2 / np.pi, downsample, fs)]
                lambda_ += dlambda_
                bw = psi_w_sigma * lambda_        
          
        while lambda_ < lambda_end:
            lambdas += [lambda_]   
            bw = psi_w_sigma * lambda_
            downsample = maximum_downsampling(self.ws, 2*bw, self.lpf_w_sigma, oversampling_factor)   
            morlets += [Morlet1DConfig(lambda_, lambda_/2 / np.pi, psi_t_sigma, psi_t_sigma/lambda_, lambda_/psi_t_sigma/2 / np.pi, 1.0/psi_t_sigma, False, 2/psi_t_sigma/2 / np.pi*lambda_, downsample, fs)]
            lambda_ *= 2**(1/Q)        

        lpf_downsample = lpf_downsampling(self.ws, self.lpf_w_sigma, oversampling_factor)
        print('LPF downsamples by ', lpf_downsample)
        #depending on the downsampling mode, modify the wavelets
        if downsample_mode == DownsampleMode.OFF:
            for m in morlets:
                m.downsampling_factor = 1
            
        elif downsample_mode == DownsampleMode.MAXIMUM_UNIFORM:
            ds = morlets[-1].downsampling_factor
            for m in morlets:
                m.downsampling_factor = ds
            
        elif downsample_mode == DownsampleMode.MULTIPLE_OF_LPF:        
            for m in morlets:
                ds = m.downsampling_factor
                #find the closest downsampling factor that will allow for a further dowmsampling step to the LPF
                while lpf_downsample % ds != 0:
                    ds -= 1                
                m.downsampling_factor = ds           
        
        #calculate the resulting sample frequency after each downsampling step
        for m in morlets:
            m.output_fs = fs/m.downsampling_factor
            print(m)        

        #prepare the DS map
        map = {}

        for i in range(len(morlets)):
            m = morlets[i]
            if m.downsampling_factor not in map.keys():
                map[m.downsampling_factor]  = [i]
            else:
                map[m.downsampling_factor] += [i]           
        
        print(map)

        ns = int(np.ceil(max(morlets[1].sigma_t*fs*approximation_support, self.lpf_t_sigma*approximation_support*fs)))
                
        self.Q: int = Q
        self.T: float = T
        self.fs: float = fs
        self.fstart: float = fstart
        self.fend: float = fend
        self.lambdas: List[float] = lambdas
        self.BW_mode: MorletBW = BW_mode
        self.scaling_mode: LambdasScalingMode = scaling_mode
        self.morlets: List[Morlet1DConfig] = morlets
        self.downsample_map: Dict[int, List[int]] = map #TODO: implement later
        self.number_of_wavelets: int = len(morlets)
        self.oversampling_factor: int = oversampling_factor
        self.max_output_BW_hz: float = morlets[-1].BW_hz
        self.min_time_support_samples: int = ns
        self.approximation_support: float = approximation_support
        self.downsample_mode = downsample_mode
        
        self.filter_downsampling_factor = self.morlets[0].downsampling_factor 
        ws_ds = self.ws/self.filter_downsampling_factor        
        self.lpf_downsampling_factor = lpf_downsampling(ws_ds, self.lpf_w_sigma, self.oversampling_factor)
        self.lpf_output_fs = self.fs / self.lpf_downsampling_factor / self.filter_downsampling_factor
        self.filter_output_fs = self.fs / self.filter_downsampling_factor
        self.eff_T = 4/self.filter_output_fs
        self.eff_lpf_w_sigma = 2 * np.pi / self.eff_T #sets the cut-off frequency to 1/T_eff Hz which modifies T to fit with the various downsampling stages
        self.eff_lpf_t_sigma = 1 / self.lpf_w_sigma
   
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
            
        #construct the LPF for each downsampling amount
        self.t_lpf = {}
        self.lpf = {}
        self.lpf_filter_size = {}
        
        for d in self.config.downsample_map.keys():            
            self.t_lpf[d] = self.t[::d]
            self.lpf[d] = morlet.sample_gauss(self.t_lpf, self.config.eff_lpf_t_sigma)
            self.lpf_filter_size[d] = len(self.lpf)
        
        # self.t_lpf = self.t[::self.config.morlets[0].downsampling_factor]
        # self.lpf = morlet.sample_gauss(self.t_lpf, self.config.eff_lpf_t_sigma)
        # self.lpf_filter_size = len(self.lpf)
        self.filter_size = len(n)
     
    
            
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
        
      
# --------------------------------------------------------------------------------------------  
# TODO: FIX THE MULTIPLE DS LEVELS, SINCE IT WONT WORK AS IT IS CURRENTLY FOR MULTIPLE LEVELS!
# --------------------------------------------------------------------------------------------  
        
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
        self.downsample_map = fb.config.downsample_map
        
        self._create_psi()
        self._create_phi()        
        
        
        print(self.Phi)
        
    def _init_psi_multiple_ds(self):
        self.Psi: Union[Dict[int, Conv1d], Conv1d] = {}
        sub_portion_found = False
        for d in self.downsample_map.keys():
            filters_in_section = self.downsample_map[d]
            if self.num_filters in filters_in_section:
                sub_portion_index = filters_in_section.index(self.num_filters)
                filters_in_section = filters_in_section[:sub_portion_index] #get rid of extraneous filters
                sub_portion_found = True
            conv = self.Psi = Conv1d(
                in_channels=1,
                out_channels=len(filters_in_section),
                kernel_size=self.fb.filter_size,
                stride=d,
                padding=self.fb.filter_size//2,
                dtype=torch_data_type,
                bias=False             
            )
            w = torch.from_numpy(self.fb.filters[:, np.newaxis, filters_in_section].T) #(N, in_channels, kernel_size)
            conv.weight = nn.Parameter(w, requires_grad = False)
            self.Psi[d] = conv
            if sub_portion_found: break
            
    def _init_psi_uniform_ds(self):
        self.Psi = Conv1d(
                in_channels=1,
                out_channels=self.num_filters,
                kernel_size=self.fb.filter_size,
                stride=self.fb.config.filter_downsampling_factor,
                padding=self.fb.filter_size//2,
                dtype=torch_data_type,
                bias=False             
            )
        w = torch.from_numpy(self.fb.filters[:, np.newaxis, 0:self.num_filters].T) #(N, in_channels, kernel_size)
        self.Psi.weight = nn.Parameter(w, requires_grad = False) 
        
    def _create_psi(self):
        if self.downsample_mode == DownsampleMode.MULTIPLE_OF_LPF:
            self._init_psi_multiple_ds()
        else:
            self._init_psi_uniform_ds()
            
    def _init_phi_multiple_ds(self):
        self.Phi = {}
        for d in self.fb.lpf.keys():
            lpf = self.fb.lpf[d]
            size = self.fb.lpf_filter_size[d]
            C = self.Psi[d].out_channels
            conv = Conv1d(
                in_channels=C,
                out_channels=C,
                kernel_size=size,
                stride=d,
                padding=size//2,
                dtype = torch_output_data_type,
                groups = C,
                bias = False         
            ) 
            w = np.tile(lpf, (self.num_filters, 1, 1))
            w = torch.from_numpy(w)
            conv.weight = nn.Parameter(w, requires_grad = False)
            self.Phi[d] = conv
    
    def _init_phi_uniform_ds(self):
        lpf = self.fb.lpf.values()[0]
        size = self.fb.lpf_filter_size.values()[0]
        self.Phi = Conv1d(
            in_channels=self.num_filters,
            out_channels=self.num_filters,
            kernel_size=size,
            stride=self.fb.config.lpf_downsampling_factor,
            padding=self.fb.lpf_filter_size//2,
            dtype = torch_output_data_type,
            groups = self.num_filters,
            bias = False         
        ) 
        w = np.tile(lpf, (self.num_filters, 1, 1))
        w = torch.from_numpy(w)
        self.Phi.weight = nn.Parameter(w, requires_grad = False)
    
    def _create_phi(self):
        if self.downsample_mode == DownsampleMode.MULTIPLE_OF_LPF:
            self._init_phi_multiple_ds()
        else:
            self._init_phi_uniform_ds()
               

    def U(self, x: torch.Tensor):
        x = x.type(torch_data_type)
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
    def __init__(self, Q_list, T, fs, oversample=1, fstart = 0.0, fend = None, ds_mode = DownsampleMode.MAXIMUM_UNIFORM) -> None:
        if not fend: fend = fs
        self.Q_list = Q_list
        self.num_levels = len(Q_list)
        self.T = T
        self.fs = fs
        self.filter_banks: List[ScatteringFB1D] = []
        #each level has a number of filterbank modules, indexed by the number of filters that should be computed
        self.filter_bank_modules: List[Dict[int, ScatteringFB1DModule]] = [] #stores a FB module that computes the number of filters specified by the key
        self.module_indices: List[List[int]] = [] #stores the number of filters required to compute the coefficients
        
        conf = ScatteringFB1DConfig(Q_list[0], T, fs, approximation_support=3.0, 
                                    oversampling_factor=oversample, fstart=fstart,         
                                    fend=fend, downsample_mode=ds_mode)
        root_fb = ScatteringFB1D(conf)
        self.filter_banks += [root_fb]
        self.module_indices += [[conf.number_of_wavelets]*conf.number_of_wavelets] #all filters in first level use the same module
        
        
        if self.num_levels > 1:
            for Q in Q_list[1:]:
                conf = ScatteringFB1DConfig(Q, T, self.filter_banks[0].config.filter_output_fs, 
                                            approximation_support=3.0, oversampling_factor=oversample, 
                                            downsample_mode=DownsampleMode.OFF)
                self.filter_banks += [ScatteringFB1D(conf)]
                
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
                for next_filter in curr_fb.config.morlets:
                    #check to see if the filter in the next filterbank will capture significant information
                    if bw < next_filter.f_c - next_filter.BW_hz: break
                    num_filts += 1
                curr_fb_amounts += [num_filts]
                if num_filts not in modules.keys():
                    modules[num_filts] = ScatteringFB1DModule(curr_fb)
                
            self.module_indices += [curr_fb_amounts]
            self.filter_bank_modules += [modules]
            
    def US1(self, x):
        return self.root_module.US(x)
    
    def _transform(self, result, u_prev, path: List[int], curr_fb_index: int):
        prev_fb_index = curr_fb_index - 1
        if curr_fb_index >= self.num_levels: return
        K = self.filter_banks[prev_fb_index].config.number_of_wavelets
        for k in range(K):
            x = u_prev[:, [k], :]
            num_filters_required = self.module_indices[curr_fb_index][k]
            module = self.filter_bank_modules[curr_fb_index][num_filters_required]
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
        
                    
                