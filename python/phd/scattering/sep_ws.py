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
from .morlet_sampler import *
from .scattering_layer import *

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
    def __init__(self, Q, T, fs, dim, N, fstart = None) -> None:
        self.Q = Q
        self.T = T
        self.fs = fs
        self.dim = dim
        self.N = N   
        Qtime = Q[0]
        Qtf = Q[1:]
        Tt = T[0]
        Ttf = T[1]

        self.ws_t = SeperableWaveletScattering([[Qtime]], [Tt], [fs], [dim], 
                    [N], include_on_axis_wavelets=False, prune=False, fstart=[fstart])
        
        self.ws_tf = SeperableWaveletScattering(
            Qtf, [Tt, Ttf], [self.ws_t.sws_layers[0].samplers[0].fs_psi_out, 1], 
            N=[self.ws_t.sws_layers[0].conv_psi.output_length[0], len(self.ws_t.sws_layers[0].samplers[0].lambdas)],
            dims=[dim, dim+1], include_on_axis_wavelets=True, prune=True)

    def scatteringTransform(self, x):
        return self.ws_tf.scatteringTransform(self.ws_t.scatteringTransform(x))
    
    def scatteringTransform(self, x: Tensor, batch_size, batch_dim, normalise=False) -> Tuple[Union[None, List[Tensor]], Tensor]:
        x = x.cuda()
        n_samples = x.shape[batch_dim]
        S_batch = []
        i = 0
        while i < n_samples:
            #only supports pruning and U discard
            xb = torch.narrow(x, batch_dim, start=i, length=min(batch_size, n_samples - i))
            ut, st = self.ws_t._US(xb, False, normalise)
            ut = ut[0]
            sf = self.ws_tf._US(ut, True, normalise).flatten(start_dim=-2)
            torch.cuda.empty_cache() 
                        
            S_batch += [torch.concat([st.cpu(), sf.cpu()], dim=-1)]
            del st, sf
            i += batch_size
            print(i)
        return torch.concat(S_batch, dim=batch_dim)

        