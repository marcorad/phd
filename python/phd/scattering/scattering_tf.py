from . import scattering_1d as s1d
from . import morlet
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple, List, Dict

from .scattering_1d import DEVICE 

from torch.nn import Conv2d

import torch.nn as nn
import torch

mpl.rcParams['text.usetex'] = True
mpl.rcParams.update(mpl.rcParamsDefault)

class ScatteringFBTFConfig:
    """
    Configure a TF scattering FB using 2 1D filterbanks. The 2D wavelet is factorised as psi(t, log(lambda)) = psi(t) * psi(log(lambda)).
    The support in quefrequency is described in terms of number of octaves to average over. If it is set to 1/Q_time, then the support is only 1 freq bin of S1.
    """
    def __init__(self, Q_time:int, Q_quefrequency, T_time, T_octaves_quefrequency, fs, approximation_support=5.0, ds_mode = s1d.DownsampleMode.MULTIPLE_OF_LPF, fs_q: float = None) -> None: 
         
        self.time_config = s1d.ScatteringFB1DConfig(Q_time, T_time, fs, 
                                              approximation_support=approximation_support, downsample_mode = ds_mode)
        
        if fs_q == None: fs_q = float(Q_time)*2.0*np.pi
        self.quefrequency_config = s1d.ScatteringFB1DConfig(Q_quefrequency, T_octaves_quefrequency, fs_q, approximation_support=approximation_support, 
                                                      downsample_mode = ds_mode)
        self.valid = self.time_config.valid and self.quefrequency_config.valid
    
    def from_S1(conf1d: s1d.ScatteringFB1DConfig, Q_time, Q_quefrequency, T_octaves_quefrequency, approximation_support=5.0):
        """
        Construct a TF Scattering config from the 1D scattering that produces the scalogram.
        """
        return ScatteringFBTFConfig(Q_time, Q_quefrequency, conf1d.T, T_octaves_quefrequency, conf1d.morlets[0].output_fs, approximation_support=approximation_support)    

 
        
class ScatteringFBTF:
    """
    Performs the sampling of the TF scattering filters for a specific configuration.
    """
    def __init__(self, config: ScatteringFBTFConfig) -> None:
        self.config = config
        self.time_fb = s1d.ScatteringFB1D(config.time_config)
        self.qf_fb_pos = s1d.ScatteringFB1D(config.quefrequency_config)
        self.qf_fb_neg = s1d.ScatteringFB1D(config.quefrequency_config, dir=-1.0)
        self._sample()        
        
    def _sample(self):
        # 2 sides of the half-plane grid
        nfilters = self.config.time_config.number_of_wavelets*self.config.quefrequency_config.number_of_wavelets*2
        # 1 side each of the filters on their respective axes
        nfilters += self.config.time_config.number_of_wavelets + self.config.quefrequency_config.number_of_wavelets
        self.filters = np.zeros((self.time_fb.filters.shape[0], self.qf_fb_pos.filters.shape[0], nfilters), dtype=s1d.FB_data_type)
        self.filter_grid_indices = {}
        self.morlet_configs = []
        
        k = 0   
        
        
        #the positive and negative sides of the quefrequency half-plane along with the time-plane
        for a in range(self.config.time_config.number_of_wavelets):
            for b in range(self.config.quefrequency_config.number_of_wavelets):
                psi_t = self.time_fb.filters[:, [a]]
                psi_log_lambda = self.qf_fb_pos.filters[:, [b]].T
                self.filters[:, :, k] =  psi_t * psi_log_lambda
                self.filter_grid_indices[k] = (a,b)                
                k += 1
                
                psi_log_lambda = self.qf_fb_neg.filters[:, [b]].T
                self.filters[:, :, k] =  psi_t * (psi_log_lambda)
                self.filter_grid_indices[k] = (a,b)
                k += 1
                
        #the filters on the time axis
        for a in range(self.config.time_config.number_of_wavelets):
            psi_t = self.time_fb.filters[:, [a]]
            phi_log_lambda = morlet.sample_gauss(self.qf_fb_pos.t, self.config.quefrequency_config.lpf_t_sigma)
            phi_log_lambda = phi_log_lambda[np.newaxis, :] #configure the transpose of the log(lambda) LPF
            self.filters[:, :, k] = psi_t * phi_log_lambda
            self.filter_grid_indices[k] = (a,-1)
            k += 1
        
        #the filters on the quefrequency axis
        for b in range(self.config.quefrequency_config.number_of_wavelets):
            psi_qf = self.qf_fb_pos.filters[:, [b]].T
            phi_t = morlet.sample_gauss(self.time_fb.t, self.config.time_config.lpf_t_sigma)
            phi_t = phi_t[:, np.newaxis] #configure the transpose of the log(lambda) LPF
            self.filters[:, :, k] = phi_t * psi_qf
            self.filter_grid_indices[k] = (-1,b)
            k += 1
            
        self.number_filters = k
        assert(k == nfilters)
        
        #construct the LPF        
        self.lpf = self.time_fb.lpf[:, np.newaxis] * self.qf_fb_pos.lpf[np.newaxis,:]            

    
    def get_morlet_properties(self, k: int) -> Tuple[float, float, float, float]:
        """
        For a filter with index k, get f_c and BW for time and quefrequency
        """
        morlets_qf = self.qf_fb_pos.config.morlets
        morlets_t = self.time_fb.config.morlets
        it, iqf = self.filter_grid_indices[k]
        fc_t = morlets_t[it].f_c if it != -1 else 0.0
        bw_t = morlets_t[it].BW_hz if it != -1 else s1d.BW_TO_SIGMA_RATIO/self.config.time_config.T
        fc_qf = morlets_qf[iqf].f_c if iqf != -1 else 0.0
        bw_qf = morlets_qf[iqf].BW_hz if iqf != -1 else s1d.BW_TO_SIGMA_RATIO/self.config.quefrequency_config.T  
        return fc_t, bw_t, fc_qf, bw_qf
    
    def plot(self, oversample = 8):
        for k in range(self.number_filters):
            psi = self.filters[:, :, k]
            Psi = fftshift(np.abs(fft2(psi, s=(psi.shape[0]*oversample, psi.shape[1]*oversample)))).T
            X = (np.arange(Psi.shape[1])/Psi.shape[1]) - 0.5 #qf
            Y = (np.arange(Psi.shape[0])/Psi.shape[0]) - 0.5 #time           
            # print(np.max(Psi))
            plt.contour(X, Y, Psi, levels=[np.max(Psi)*0.6])
            
        
        plt.xlabel("$\omega_t$")
        plt.ylabel("$\omega_{log \lambda}$")
        # plt.xlim([0,0.5])
        
        print(self.number_filters)   
        
        plt.show(block=True)
        
    def plot_wavelet(self, k):
        psi = self.filters[:, :, k].T
        plt.subplot(311)
        plt.imshow(np.real(psi), aspect='equal')
        plt.subplot(312)
        plt.imshow(np.imag(psi), aspect='equal')
        plt.subplot(313)
        plt.imshow(np.abs(psi), aspect='equal')
        plt.show(block=True)
        
    def plot_lpf(self):
        plt.imshow(self.lpf.T)
        plt.xlabel('Time')
        plt.ylabel('Quefrequency')
        plt.show(block=True)
        
        
def get_eligible_filter_indices(fb0: ScatteringFBTF, k0 , fb1: ScatteringFBTF) -> Tuple[int]:    
        """
        For a given filter with index k0 in a FB fb0, compute all the relevant filter indices 
        in fb1 that will capture significant information.
        """          
        _, curr_filter_bw_t, _, curr_filter_bw_qf = fb0.get_morlet_properties(k0)
        
        indices = []
        
        for k in range(len(fb1.filter_grid_indices)):
            fct1, bwt1, fcqf1, bwqf1 = fb1.get_morlet_properties(k)
            EPS = 1e-6 #TODO: make EPS adaptive to the smallest BW
            #check to see if the current filter will contain energy in frequency support
            if curr_filter_bw_t  < fct1 - bwt1*s1d.BW_TO_SIGMA_RATIO + EPS: continue
            if curr_filter_bw_qf < fcqf1 - bwqf1*s1d.BW_TO_SIGMA_RATIO + EPS: continue
            #check passed, so add it to the eligible filter index list
            indices += [k]
            
        return tuple(indices)
    
class ScatteringFBTFModule(nn.Module):
    '''
    The torch module that performs the scattering computations for a single 1D filter bank.
    '''
    def __init__(self, fb: ScatteringFBTF, filter_indices = None) -> None:
        super().__init__()
        self.fb = fb
        if filter_indices == None: filter_indices = [i for i in range(len(fb.filter_grid_indices))]
        self.filter_indices = filter_indices
        
        #for convenience        
        self.downsample_mode = fb.config.time_config.downsample_mode
        
        
        
        self._create_psi()
        self._create_phi() 
        
    def _create_psi(self):
        psi = self.fb.filters
        size = psi.shape
        Lqf = size[1]
        Lt = size[0]
        n_filters = len(self.filter_indices)
        self.Psi = Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(Lqf, Lt),
            stride=[self.fb.config.quefrequency_config.filter_downsampling_factor, self.fb.config.time_config.filter_downsampling_factor],
            padding=(Lqf//2, Lt//2),
            bias=False,
            groups=1,
            dtype=s1d.TORCH_DATA_TYPE,   
            device = DEVICE   
        )
        #(St, Sqf, filters) -> (filters, Sqf, St)
        psi = np.swapaxes(psi, 0, 2)
        psi = psi[list(self.filter_indices), :, :]
        Lt = torch.from_numpy(psi[:, np.newaxis, :, :]) #(out_channels, in_channels/groups, w, h)
        Lt = Lt.to(DEVICE)
        self.Psi.weight = nn.Parameter(Lt, requires_grad=False)
    
    def _create_phi(self):
        lpf = self.fb.lpf
        size = lpf.shape
        Lqf = size[1]
        Lt = size[0]
        n_filters = len(self.filter_indices)
        #TODO, check the orientation of the ScatteringFB1D output
        self.Phi = Conv2d(
            in_channels=n_filters,
            out_channels=n_filters,
            groups=n_filters,
            kernel_size=(Lqf, Lt),
            stride=[self.fb.config.quefrequency_config.lpf_downsampling_factor, self.fb.config.time_config.lpf_downsampling_factor],
            padding=(Lqf//2, Lt//2),
            bias=False,
            dtype=s1d.TORCH_OUTPUT_DATA_TYPE, 
            device = DEVICE  
        )
        #(1, St, Sqf) -> (1, Sqf, St)
        lpf = lpf[np.newaxis, :, :]
        lpf = np.swapaxes(lpf, 1, 2)
        #(1, St, Sqf) -> (n_filters, 1, Sqf, St)
        lpf = np.tile(lpf, (n_filters, 1, 1, 1))
        print("LPF SHAPE ", lpf.shape)
        Lt = torch.from_numpy(lpf) #(out_channels, in_channels/groups, w, h)
        Lt = Lt.to(DEVICE)
        self.Phi.weight = nn.Parameter(Lt, requires_grad=False)
    
    #All functions expect an input of size (N_batch, 1, L_qf, L_time)    
        
    def U(self, x: torch.Tensor):
        x = x.type(s1d.TORCH_DATA_TYPE)
        return torch.abs(self.Psi(x))        
        
    def S(self, u: torch.Tensor):
        return self.Phi(u)
    
    def US(self, x: torch.Tensor):
        u = self.U(x)
        s = self.S(u)
        return u, s
        
        
class ScatteringTF:
    def __init__(self, Q_s1, T, T_octaves_quefrequency, fs, Q_tf = [(1, 1)], oversample=1, fstart=0.0, fend=None) -> None:          
        
        self.scattering_s1 = s1d.Scattering1D([Q_s1], T, fs, oversample, fstart, fend)
        self.T = self.scattering_s1.T #may be different than specified for optimization
        self.num_tf_levels = len(Q_tf)
        
        self.Q_tf = Q_tf
        self.tf_filter_banks: List[ScatteringFBTF] = []
        #each level has a number of filterbank modules, indexed by the number of filters that should be computed
        self.tf_filter_bank_modules: List[Dict[Tuple[int], ScatteringFBTF]] = [] #stores a FB module that computes the number of filters specified by the key
        self.tf_module_indices: List[List[Tuple[int]]] = [] #stores the number of filters required to compute the coefficients
        self.fs_audio = fs
        self.fs_scal = self.scattering_s1.filter_banks[0].config.filter_output_fs
        
        #set fs = Qs, sigma_n = sigma_t*fs = Toct*Qs -> T/2pi*Qs=Toct*Qs -> T=Toct*2pi
        conf = ScatteringFBTFConfig(Q_tf[0][0], Q_tf[0][1], self.T, T_octaves_quefrequency*np.pi*2, self.fs_scal,
                                    approximation_support=3.0, ds_mode=s1d.DownsampleMode.MULTIPLE_OF_LPF, fs_q=Q_s1)
        self.T_qf = conf.quefrequency_config.T #may be different than specified for optimization
        root_fb = ScatteringFBTF(conf)
        self.tf_filter_banks += [root_fb]
        self.tf_module_indices += [[]] #this will be unused anyways
        
        self.root_module = ScatteringFBTFModule(root_fb)
        self.tf_filter_bank_modules += [{}] #will be unused anyways
        
        prev_level = 0
        #prepare the filterbank samplers
        if self.num_tf_levels > 1:
            for Q_t, Q_qf in Q_tf[1:]:
                conf = ScatteringFBTFConfig(Q_t, Q_qf, self.T, self.T_qf, 
                                            self.tf_filter_banks[prev_level].config.time_config.filter_output_fs,
                                            approximation_support=3.0, ds_mode=s1d.DownsampleMode.MULTIPLE_OF_LPF, 
                                            fs_q=self.tf_filter_banks[prev_level].config.quefrequency_config.filter_output_fs)
                if not conf.valid:
                    print(f'ONLY CONFIGURING FOR {prev_level + 1} LEVELS.')
                    self.num_tf_levels = prev_level + 1
                    self.Q_tf = self.Q_tf[:self.num_tf_levels]
                    break
                
                self.tf_filter_banks += [ScatteringFBTF(conf)]
                prev_level += 1
        
        
        #prepare the torch modules
        for k in range(1, self.num_tf_levels):
            prev_fb = self.tf_filter_banks[k-1]
            curr_fb = self.tf_filter_banks[k]
            modules = {}
            indices = []
            
            for curr_filter_index in range(prev_fb.number_filters):
                eligible_indices = get_eligible_filter_indices(prev_fb, curr_filter_index, curr_fb)
                if eligible_indices not in modules.keys():
                    modules[eligible_indices] = ScatteringFBTFModule(curr_fb, eligible_indices) if len(eligible_indices) > 0 else None
                indices += [eligible_indices]
            
            self.tf_module_indices += [indices]
            self.tf_filter_bank_modules += [modules]
            
    def _transform(self, result, u_prev, path: List[int], curr_fb_index: int):
        if curr_fb_index >= self.num_tf_levels: return
        K = u_prev.shape[1]
        for k in range(K):
            num_filters_required = self.tf_module_indices[curr_fb_index][k]
            module = self.tf_filter_bank_modules[curr_fb_index][num_filters_required]
            if module == None: continue
            x = u_prev[:, [k], :, :]
            u_curr, s_curr = module.US(x)
            p = path + (k, )
            result[p] = (u_curr, s_curr, u_curr.shape[1], u_curr.shape[2], u_curr.shape[3])
            self._transform(result, u_curr, p, curr_fb_index+1)
            
    def US1(self, x):
        return self.root_module.US(x)
            
    def transform(self, x: torch.Tensor):
        result = {}
        #create the S1 scattering
        u_scal, s_scal = self.scattering_s1.US1(x)
        #create the first TF level
        path = (0,)
        #(N_batch, L_qf, L_t) -> (N_batch, 1, L_qf, L_t)
        x_in = u_scal[:, None, :, :] #(N_batch, 1, L_qf, L_t)
        result[(-1,)] = (x_in, s_scal[:, None, :, :], 1, u_scal.shape[1], u_scal.shape[2])
        u_curr, s_curr = self.US1(x_in)
        result[path] = (u_curr, s_curr, u_curr.shape[1], u_curr.shape[2], u_curr.shape[3])
        #transform for subsequent TF levels
        self._transform(result, u_curr, path, 1)
        return result
    
    def get_slope(self, level:int, index:int):
        fct, _, fcqf, _ = self.tf_filter_banks[level].get_morlet_properties(index)
        return fct, fcqf
    
    def feature_vector(self, result):
        n_feats = 0
        for v in result.values():
            n_feats += v[0].shape[1]*v[0].shape[2]
        print(n_feats)
                
        
                
                
            
            