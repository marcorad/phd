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
    
    def __init__(self, Q, T, fs, dims, N, include_on_axis_wavelets=True, fstart = None) -> None:
        self.J = len(Q)
        self.dims = dims
        self.Q = Q
        self.T = T
        self.fs = fs
        self.N = N
        self.include_on_axis_wavelets = include_on_axis_wavelets
        self.fstart = fstart
        self.s0_layer = SeperableScatteringLayerS0(T, fs, dims, N)        
        
        #recursively init all the required filters
        self.sws_layers: Dict[Tuple, SeperableScatteringLayer] = {}
        self.sws_layers[((0,), (0,))] = SeperableScatteringLayer(self.Q[0], self.T, self.fs, self.dims, self.N, include_on_axis_wavelets=self.include_on_axis_wavelets, fstart=self.fstart)        
        self.n_paths = len(self.sws_layers[((0,), (0,))].paths)
        self._init_sws_layers((0,), 1, self.sws_layers[((0,), (0,))])    
        print(f'SWS configuration has {self.n_paths} paths.')
        
    def _init_sws_layers(self, prev_path: Tuple, depth, prev_layer: SeperableScatteringLayer):
        if depth == self.J: return
        for path_idx, filters in enumerate(prev_layer.psi.values()):
            fs = []
            N = []
            fc = []
            lambdas = []
            bw_lim_w = []
            for i, f in enumerate(filters):
                fs.append(f.fs_out)  
                N.append(ceil(prev_layer.N[i] / f.ds_lambda))  
                fc.append(f.f_c)
                bw_lim_w.append(f.bw_w)
                lambdas.append(f.lambda_)
            curr_path = prev_path + (path_idx,)
            key = (curr_path, tuple(lambdas))
            print(f'Initialising SWS @ {key}, freq={fc}, N={N}, fs={fs}')            
            sws = SeperableScatteringLayer(self.Q[depth], self.T, fs, self.dims, N, include_on_axis_wavelets=self.include_on_axis_wavelets, bw_lim_w=bw_lim_w)
            if(len(sws.paths) > 0): #only add this layer if there are filters inside it
                self.sws_layers[key] = sws
                self.n_paths += len(sws.paths)
                self._init_sws_layers(curr_path, depth+1, sws)    
        


    def _US_rec(self, U: Dict[Tuple, Dict[Tuple, Tensor]], S, prev_path: Tuple, depth, discard_U = True):
        if depth == self.J: return     
        u_curr = U[prev_path]
        for i, curr_lambda in enumerate(u_curr.keys()):
            curr_path = prev_path + (i,)            
            key: Tuple = (curr_path, curr_lambda)
            if key in self.sws_layers.keys():
                curr_layer = self.sws_layers[key]
                u_curr_lambda = u_curr[curr_lambda]                
                u, s = curr_layer.US(u_curr_lambda)
                if discard_U: del u_curr_lambda
                S[curr_path] = s      
                U[curr_path] = u      
                self._US_rec(U, S, curr_path, depth+1)  
    
    def _US(self, x: Tensor, discard_U = True, concat_s = True) -> Union[Tuple[List[Tensor], Tensor], Tensor]:        
        s0 = self.s0_layer.S0(x)
        u1, s1 = self.sws_layers[((0,), (0,))].US(x) 
        U = {}
        S = {}
        S[(-1,)] = {(-1,): s0}
        U[(0,)] = u1
        S[(0,)] = s1
        self._US_rec(U, S, (0,), 1, discard_U=discard_U) 
        if concat_s: 
            s_list = []
            for s_all in S.values():
                for s in s_all.values():
                    s_list.append(s.unsqueeze(-1))
            s = torch.concat(s_list, dim=-1)       
        return U, s if concat_s else S
    
    def scatteringTransform(self, x: Tensor, batch_size = None, batch_dim = None, discard_U = True, prune = True, flatten = False, dct = False, normalise=False) -> Tuple[Union[None, List[Tensor]], Tensor]:
        x = x.cuda()
        if batch_dim == None:
            U, S = self._US(x, discard_U, concat_s=True)  
            if discard_U: return S
            else: return U, S 
        else:
            n_samples = x.shape[batch_dim]
            S_batch = []
            i = 0
            while i < n_samples:
                #only supports pruning and U discard
                xb = torch.narrow(x, batch_dim, start=i, length=min(batch_size, n_samples - i))
                u, s = self._US(xb, True, concat_s=True)
                torch.cuda.empty_cache()                             
                S_batch += [s.cpu()]
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

        self.ws_t = SeperableScatteringLayer(Qtime, Tt, [fs], dim[0], N, include_on_axis_wavelets=False, fstart=fstart, allow_seperate_ds=False)
        
        self.ws_tf = SeperableWaveletScattering(
            Qtf, [Tt, Ttf], [self.ws_t.sws_layers[0].samplers[0].fs_psi_out, 1], 
            N=[self.ws_t.conv_psi.values()[0].output_length[0], len(self.ws_t.samplers[0].lambdas)],
            dims=[dim, dim+1], include_on_axis_wavelets=True)

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

        