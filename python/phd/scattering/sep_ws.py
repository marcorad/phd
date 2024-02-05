# seperable wavelet scattering

from dataclasses import dataclass
from typing import List, Dict, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from sympy.ntheory import factorint

from .morlet import sample_gauss, sample_morlet
from .conv import SeperableConv
from .config import *

from itertools import product
from math import ceil, floor

from tqdm import tqdm

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
        while sigma_phi_w > sigma_lambda_w and lambda_ < ws/2 - sigma_phi_w*MORLET_DEFINITION.beta:
            lambdas += [lambda_]
            psi += [sample_morlet(t, lambda_, 1/sigma_phi_w*lambda_, dir = self.polarity).astype(NUMPY_COMPLEX) / self.fs_in]
            lambda_ += MORLET_DEFINITION.alpha(self.Q)*sigma_phi_w
            sigmas += [sigma_phi_w]
            sigma_lambda_w = lambda_*sigma_w

        #build exponential filters for all filters until the limit is reached
        while lambda_ < ws/2 - sigma_lambda_w*MORLET_DEFINITION.beta:
            lambdas += [lambda_]
            psi += [sample_morlet(t, lambda_, 1/sigma_w, dir = self.polarity).astype(NUMPY_COMPLEX)  / self.fs_in]
            lambda_ *= 2**(1/self.Q)
            sigmas += [sigma_lambda_w]
            sigma_lambda_w = lambda_*sigma_w

        assert len(lambdas) > 0, f"The filterbank requested cannot be constructed.\nFor an invariance scale of {self.T} (fc = {sigma_phi_w/2/PI} Hz), the current sample frequency of {self.fs_in} cannot support more filters."

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

        #if polarity is negative, reverse the lists to maintain small-to-large order
        if self.polarity < 0:
            self.lambdas.reverse()            
            self.bw_w.reverse()
            self.bw_f.reverse()
            self.fc.reverse()
            self.psi = np.flip(self.psi, 0)

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
    def __init__(self, Q, T, fs_in, include_lpf = True, fstart = None) -> None:
        self.fb_pos = MorletSampler1D(Q, T, fs_in, pol=+1, include_lpf=include_lpf, fstart = fstart) #include lpf in positive fb
        self.fb_neg = MorletSampler1D(Q, T, fs_in, pol=-1, fstart = fstart)

        #properties from MorletSampler1D
        self.Q = Q
        self.T = T
        self.fs_in = fs_in
        self.d_tot = self.fb_neg.d_tot
        self.d_lambda = self.fb_neg.d_lambda
        self.d_phi = self.fb_pos.d_phi
        self.fs_psi_out = self.fb_neg.fs_psi_out
        self.fs_out = self.fb_neg.fs_out
        self.lambdas = [-l for l in self.fb_neg.lambdas] + self.fb_pos.lambdas 
        self.fc = [l/2/PI for l in self.lambdas]
        self.sigma_phi_w  = self.fb_neg.sigma_phi_w
        self.bw_w = self.fb_neg.bw_w + self.fb_pos.bw_w
        self.bw_f = [b / 2 / PI for b in self.bw_w]  

        self.psi = np.concatenate((self.fb_neg.psi, self.fb_pos.psi), axis=0, dtype=NUMPY_COMPLEX)
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
                 fs_in: Union[List[float], float], dims: Union[List[int], int], N: Union[List[int], int],
                 include_on_axis_wavelets = True, fstart: Union[List[float], None] = None) -> None:
        
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
          

        #first dimension is only half of the frequency plane, since we expect input to be real
        #NOTE: when convolving, take care to exclude the path where the filter is at 0 (in d dimensions)
        self.samplers = [MorletSampler1D(Q[0], T[0], fs_in[0], include_lpf=include_on_axis_wavelets, fstart= fstart[0] if fstart else None)]
        self.fzero_idx = [0]
        for i in range(1, self.ndim):
            samp = MorletSampler1DFull(Q[i], T[i], fs_in[i], include_lpf=include_on_axis_wavelets, fstart = fstart[i] if fstart else None)
            self.samplers += [samp]
            self.fzero_idx += [len(samp.lambdas)//2]

        self.conv_psi = SeperableConv(
            [torch.from_numpy(s.psi) for s in self.samplers], 
            N, [s.d_lambda for s in self.samplers], dims, enable_freq_ds=ENABLE_FREQ_DS
            )
        self.conv_phi = SeperableConv(
            [torch.from_numpy(s.phi) for s in self.samplers], 
            [ceil(n/s.d_lambda) for n, s in zip(N, self.samplers)], 
            [s.d_phi for s in self.samplers], 
            dims, enable_freq_ds=ENABLE_FREQ_DS
            )           

    
    def select_filters(self, bw_w = None, prune = True):

        if bw_w == None: 
            bw_w = [fs*PI for fs in self.fs_in]                    

        #select the filters based on the input BW
        filter_idx = {}
        for i in range(self.ndim):
            sampler = self.samplers[i]
            start = -1
            start_lambda = -1
            if prune:
            #look for the first filter that falls within the BW
                for j in range(len(sampler.lambdas)):
                    l, bw = sampler.lambdas[j], sampler.bw_w[j]
                    if abs(l) <= bw_w[i]:
                        start = j
                        start_lambda = l
                        break
            else:
                start = 0
                start_lambda = sampler.lambdas[0]
            #if none were found, ignore this filterbank
            if start == -1: 
                return None
            
            #look for the first filter that exceeds the BW
            end = -1
            end_lambda = -1
            if prune:
                for j in range(start, len(sampler.lambdas)):
                    l, bw = sampler.lambdas[j], sampler.bw_w[j]
                    if abs(l) > bw_w[i]:
                        end = j - 1
                        end_lambda = sampler.lambdas[j - 1]
                        break
            else: 
                end = len(sampler.lambdas) - 1
                end_lambda = sampler.lambdas[end]

            #if none were found, then all filters must be used
            if end == -1:
                end = len(sampler.lambdas) - 1
                end_lambda = sampler.lambdas[-1]
            length = end - start + 1

            #ignore this filterbank if it is the 0 filter (in the case of include_on_axis_wavelets = True)
            if start_lambda == 0 and end_lambda == 0: return None


            #add the metadata to the list
            filter_idx[self.conv_dims[i]] = {
                'start': start,
                'length': length,
                'end': end,
                #debug
                'start_lambda': start_lambda,
                'end_lambda': end_lambda,
                'bw': bw_w[i],
                'end_lambda_bw': sampler.bw_w[end],
                'lambdas': sampler.lambdas[start:end+1]
            }

        #get the BWs of the flattened output
        selected_bws = []
        for i in range(self.ndim):
            idx = filter_idx[self.conv_dims[i]]
            bws = self.samplers[i].bw_w[idx['start']:idx['end']+1]
            selected_bws.append(bws)
        

        return filter_idx, list(product(*selected_bws))
    

    def _conv_psi(self, x: Tensor, filter_idx):
        if filter_idx == None: return None        
        psi = self.conv_psi.convolve(x, filter_idx, fun_before_ds=torch.abs)
        s = psi.shape[:len(x.shape)]
        return psi.reshape(*s, -1)  

    def _conv_phi(self, x: Tensor):        
        phi = self.conv_phi.convolve(x, fun_before_ds=torch.real) 
        s = phi.shape[:len(x.shape)]           
        return phi.squeeze([-n for n in range(1, len(x.shape)+1)])
    
    def US(self, x: Tensor, filter_idx = None):
        x = self.conv_psi.add_padding(x, pad_mode='constant')
        U = self._conv_psi(x, filter_idx)
        U = self.conv_psi.remove_padding(U)
        if U == None: return None, None
        Up = self.conv_phi.add_padding(U, pad_mode='constant')
        Sp = self._conv_phi(Up)
        S = self.conv_phi.remove_padding(Sp)
        del Up, Sp
        return U, S    
    
    def get_psi_output_fs(self):
        return [s.fs_psi_out for s in self.samplers]

    def get_psi_output_N(self):
        return [ceil(n/s.d_lambda) for n, s in zip(self.N, self.samplers)]
    

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
            phi = phi[np.newaxis, :] #rehape to (1, L) 
            d_tot = max(int(np.floor(fs_in[i] * T[i] / 2 / MORLET_DEFINITION.beta)), 1) if ENABLE_DS else 1 #total allowed downsampling 
            self.d_tot += [d_tot]
            self.phi += [torch.from_numpy(phi)]

        self.conv = SeperableConv(self.phi, N, self.d_tot, self.conv_dims, enable_freq_ds=ENABLE_FREQ_DS)

        
    def S0(self, x: Tensor):
        x = self.conv.add_padding(x, pad_mode="constant")
        s0p = self.conv.convolve(x, fun_before_ds=torch.real)
        s0 = self.conv.remove_padding(s0p).squeeze([-n for n in range(1, self.ndim)])
        del s0p
        return s0
        

class SeperableWaveletScattering:
    
    def __init__(self, Q, T, fs, dims, N, include_on_axis_wavelets=True, prune=True, fstart = None) -> None:
        self.J = len(Q)
        self.dims = dims
        self.Q = Q
        self.T = T
        self.fs = fs
        self.prune = prune
        self.N = N
        self.s0_layer = SeperableScatteringLayerS0(T, fs, dims, N)
        
        self.sws_layers: List[SeperableScatteringLayer] = [SeperableScatteringLayer(Q[0], T, fs, dims, N, include_on_axis_wavelets, fstart)]
        for j in range(1, self.J):
            self.sws_layers += [SeperableScatteringLayer(
                Q[j], T, self.sws_layers[j-1].get_psi_output_fs(), dims, 
                self.sws_layers[j-1].get_psi_output_N(), include_on_axis_wavelets
                )]
            
        self._compute_paths()
        self._compute_SU_metadata()
        self._remove_root_wavelet()


    def _compute_paths_rec(self, paths: Dict, curr_layer: int, curr_bws: List, prev_path_index: Tuple):
        if curr_layer >= len(self.sws_layers): return
        ws = self.sws_layers[curr_layer]
        for i, bw in enumerate(curr_bws):
            idx, next_bws = ws.select_filters(bw, self.prune)
            
            curr_path_index = prev_path_index + (i,)
            paths[curr_path_index] = idx
            self._compute_paths_rec(paths, curr_layer+1, next_bws, curr_path_index)            

    def _compute_paths(self):
        paths = {}
        idx, bws = self.sws_layers[0].select_filters(prune=self.prune)
        
        paths[(0,)] = idx
        self._compute_paths_rec(paths, 1, bws, (0,))  
        self.paths = paths  

    def _compute_SU_metadata(self):
        self.path_lambdas = {}
        for k, v in self.paths.items():
            lambdas_list = []
            for dim, path in v.items():
                lambdas_list.append(path['lambdas'])
            self.path_lambdas[k] = list(product(*lambdas_list))
        self.flattened_lambdas = []
        for k, v in self.path_lambdas.items():
            self.flattened_lambdas += v
        

    def _remove_root_wavelet(self):
        new_flat_l = []
        self.keep_idx: List[int] = []
        for i, ls in enumerate(self.flattened_lambdas):
            if not all([l == 0 for l in ls]): 
                new_flat_l.append(ls)
                self.keep_idx.append(i + 1) #account for S0 offset
        self.flattened_lambdas = new_flat_l   
        new_path_l = {}
        linear_path_lambdas = []
        for k, v in self.path_lambdas.items():
            new_ls = []
            for ls in v:
                if not all([l == 0 for l in ls]): 
                    new_ls.append(ls)
            new_path_l[k] = new_ls
            linear_path_lambdas.append(new_ls)
        self.path_lambdas = new_path_l
        self.path_lambdas[(-1,)] = tuple([0 for _ in range(len(self.dims))])
        self.linear_path_lambdas = linear_path_lambdas
        


    def _US_rec(self, U, S, prev_path, curr_level: int, u_curr: Tensor, discard_U, normalise):
        if curr_level == self.J: return        
        sws = self.sws_layers[curr_level]
        n_filt = u_curr.shape[-1]
        U_temp = []
        for n in range(n_filt):
            u_n = u_curr.select(dim=-1, index=n)
            curr_path = prev_path + (n,)
            u, s = sws.US(u_n, self.paths[curr_path])   
            if normalise: s = self._normalise(s)         
            if u != None:
                S[curr_path] = s
                self._US_rec(U, S, curr_path, curr_level+1, u, discard_U, normalise)
                if not discard_U: U_temp += [u]
                else: del u
        if not discard_U: U += U_temp

    def _normalise(self, s: Tensor):
        smax = s
        for d in self.dims:
            smax, _ = torch.max(smax, dim=d, keepdim=True)        
        smax, _ = torch.max(smax, dim=-1, keepdim=True)
        # smax[smax <= 1e-12] = 1.0
        sdiv = s/smax
        del smax, s
        return sdiv

    
    def _US(self, x: Tensor, discard_U = True, normalise=False) -> Union[Tuple[List[Tensor], Tensor], Tensor]:        
        s0 = self.s0_layer.S0(x)
        u1, s1 = self.sws_layers[0].US(x, self.paths[(0,)])  
        if normalise:
            s1 = self._normalise(s1)     
            s0 = self._normalise(s0)     
        U = None if discard_U else [u1]
        S = {}
        S[(-1,)] = s0
        S[(0,)] = s1
        self._US_rec(U, S, (0,), 1, u1, discard_U, normalise) 
        s_list = list(S.values())       
        S = torch.concat(s_list, dim=-1).cpu()
        S = torch.index_select(S, dim=-1, index=torch.IntTensor([0] + self.keep_idx))
        if discard_U:
            del u1 
            return S
        else:
            return U, S
    
    def scatteringTransform(self, x: Tensor, batch_size = None, batch_dim = None, discard_U = True, prune = True, flatten = False, dct = False, normalise=False) -> Tuple[Union[None, List[Tensor]], Tensor]:
        x = x.cuda()
        if batch_dim == None:
            return self._US(x, discard_U, normalise)            
        else:
            n_samples = x.shape[batch_dim]
            S_batch = []
            i = 0
            while i < n_samples:
                #only supports pruning and U discard
                xb = torch.narrow(x, batch_dim, start=i, length=min(batch_size, n_samples - i))
                s = self._US(xb, True, normalise).cpu()
                torch.cuda.empty_cache() 
                            
                S_batch += [s]
                i += batch_size
                print(i)
            return torch.concat(S_batch, dim=batch_dim)
        
    def _dct(self, S: Tensor, d):
        orig_shape = S.shape
        N = S.shape[d]
        S = torch.cat([S, torch.slice_copy(torch.flip(S, dims = [d]), start=1, dim = d)], dim=d)
        S = torch.view_as_real(torch.fft.rfft(S, dim=d))
        k = - torch.arange(N, dtype=S.dtype, device=S.device) * np.pi / (2 * N)
        new_shape = np.ones(len(S.shape)-1, dtype=np.int64).tolist()
        new_shape[d] = N
        k = k.reshape(new_shape)
        W_r = torch.cos(k)
        W_i = torch.sin(k)
        S = S[..., 0] * W_r - S[..., 1] * W_i
        S = 2 * S.view(*orig_shape)
        return S
                


class JointTFScattering:
    def __init__(self, Q, T, fs, dim, N) -> None:
        self.Q = Q
        self.T = T
        self.fs = fs
        self.dim = dim
        self.N = N   
        Qtime = Q[0]
        Qtf = Q[1:]
        Tt = T[0]
        Ttf = T[1]

        #TODO: add frequency limits!

        self.ws_t = SeperableWaveletScattering([Qtime], [Tt], [fs], dim, 
                    N, include_on_axis_wavelets=False, prune=False)
        
        self.ws_tf = SeperableWaveletScattering(
            Qtf, [Tt, Ttf], [self.ws_t.sws_layers[0].samplers[0].fs_psi_out, 1], 
            dims=[dim, dim+1], include_on_axis_wavelets=True, prune=True)

    def scatteringTransform(self, x):
        return self.ws_tf.scatteringTransform(self.ws_t.scatteringTransform(x))

        