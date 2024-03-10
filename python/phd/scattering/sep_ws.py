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
import sys

PI = np.pi

def optimise_T(T_nom, fs, eps = 0.1):
    beta = MORLET_DEFINITION.beta
    d1 = int(np.floor(fs * T_nom * (1-eps) / 2 / beta))
    d2 = int(np.floor(fs * T_nom * (1+eps) / 2 / beta))
    kmax = -1
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
    
    def __init__(self, Q, T, fs, dims: List[int], N: List[int], include_on_axis_wavelets=True, fstart = None) -> None:
        
        #normalise for numerical stability (some combinations my combinations may cause precision errors that propogate)
        T = [f*t for f, t in zip(fs, T)]
        if fstart != None: fstart = [s/f for f, s in zip(fs, fstart)]
        fs = [1 for _ in fs]
        
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
        # print(f'SWS configuration has {self.n_paths} paths.')
        
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
            # print(f'Initialising SWS @ {key}, freq={fc}, N={N}, fs={fs}')            
            sws = SeperableScatteringLayer(self.Q[depth], self.T, fs, self.dims, N, include_on_axis_wavelets=self.include_on_axis_wavelets, bw_lim_w=bw_lim_w)
            if(len(sws.paths) > 0): #only add this layer if there are filters inside it
                assert(key not in self.sws_layers.keys())
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
                
    def _concat_s(self, S):
        s_list = []
        for s_all in S.values():
            if type(s_all) == Dict:
                for s in s_all.values():
                    s_list.append(s.unsqueeze(-1))
            elif type(s_all) == Tensor:
                s_list.append(s_all)
        s = torch.concat(s_list, dim=-1) 
        return s
    
    def _compute_dct(self, ws: Dict[Tuple, Tensor]) -> Tensor:   
        ndim = len(self.dims)     
        lambdas = [set() for _ in range(ndim)] #each dimension has a set of lambdas
        for lds in list(ws.keys()):
            for i, l in enumerate(lds):
                lambdas[i].add(l)
        for i in range(len(lambdas)): lambdas[i] = sorted(lambdas[i]) #sort these lambdas
        idx_dict = [{l: i for i, l in enumerate(s)} for s in lambdas] #each lambda is then associated with an index
        ws0 = list(ws.values())[0] #get the output shape
        pad_shape = list(ws0.shape) + [len(s) for s in lambdas]
        y = torch.zeros(pad_shape, dtype=ws0.dtype, device=ws0.device) #initialise the output
        for k, v in ws.items():
            idx = [idx_dict[n][l] for n, l in enumerate(k)] #get the indices associated with a specific lambda  
            y[..., *idx] = v #copy them into the output
                
            
        #create type-II fft boundary reflection
        for i in range(ndim):
            y = torch.concat((y, torch.flip(y, dims=[-1-i])), dim=-1-i)
            
        #compute DCT over each dimension
        for i in range(ndim):
            d = -ndim + i
            N = y.shape[d]
            y = torch.fft.rfft(y, dim=d).narrow(start = 0, length=N//2, dim=d)
            k = torch.arange(N//2)
            w = torch.exp(-1j * torch.pi * k / N) #linear phase ramp
            #add dimensions before
            for _ in range(len(ws0.shape) + i): w = w.unsqueeze(0)
            #add dimensions after
            for _ in range(ndim-i-1): w = w.unsqueeze(-1)
            #now it will properly broadcast
            y = torch.real(y * w)
        return y.reshape(*ws0.shape, -1) #return type-II DCT flattened
    
    def _dct(self, S):
        S_dct = {}
        for p, s in S.items():
            if p[0] != -1:
                S_dct[p] = self._compute_dct(s)
            else:
                S_dct[p] = s
        return S_dct
    
    def _US(self, x: Tensor, discard_U = True) -> Union[Tuple[List[Tensor], Tensor], Tensor]:        
        s0 = self.s0_layer.S0(x)
        u1, s1 = self.sws_layers[((0,), (0,))].US(x) 
        U = {}
        S = {}
        S[(-1,)] = {(-1,): s0}
        U[(0,)] = u1
        S[(0,)] = s1
        self._US_rec(U, S, (0,), 1, discard_U=discard_U)    
        return U, S
    
    def scatteringTransform(self, x: Tensor, batch_size = None, batch_dim = None, discard_U = True, dct=False) -> Tuple[Union[None, List[Tensor]], Tensor] | Tensor:
        x = x.cuda()
        if batch_dim == None:
            U, S = self._US(x, discard_U) 
            if dct: S = self._dct(S)
            S = self._concat_s(S) 
            if discard_U: return S
            else: return U, S 
        else:
            n_samples = x.shape[batch_dim]
            S_batch = []
            i = 0
            with tqdm(total=n_samples) as pbar:
                while i < n_samples:
                    #only supports pruning and U discard
                    l = min(batch_size, n_samples - i)
                    xb = torch.narrow(x, batch_dim, start=i, length=l)
                    u, s = self._US(xb, True)                    
                    del u
                    if dct: s = self._dct(s)
                    s = self._concat_s(s)
                    torch.cuda.empty_cache()                             
                    S_batch += [s.cpu()]
                    i += batch_size
                    pbar.update(l)
            return torch.concat(S_batch, dim=batch_dim)
        
    def get_s_sample_freq(self) -> List[float]:
        sampl = list(self.sws_layers.values())[0].samplers
        return [s.fs_out for s in sampl]
    
    def get_total_ds(self) -> List[float]:
        sampl = list(self.sws_layers.values())[0].samplers
        return [s.d_tot for s in sampl]     


class JointTFScattering:
    def __init__(self, Qtime, Qtf, Ttime, Ttf, fs, dim, N, fstart = None) -> None:
        self.Qtime = [Qtime]
        self.Qtf = Qtf
        self.Ttime = [Ttime]
        self.Ttf = Ttf
        self.fs = [fs]
        self.dim = [dim]
        self.N = [N]   
        self.ws_t = SeperableScatteringLayer(Qtime, Ttime, fs, dim, N, include_on_axis_wavelets=False, fstart=[fstart], allow_seperate_ds=False)
        
        fs_psi_time_out = self.ws_t.samplers[0].psi[0].fs_out    
        Nt = list(self.ws_t.conv_psi.values())[0].output_length[0]
        # print([v.output_length for v in list(self.ws_t.conv_psi.values())])
        Nf = len(self.ws_t.samplers[0].psi)       
        
        
        self.ws_tf = SeperableWaveletScattering(
            Qtf, [Ttime[0], Ttf], [fs_psi_time_out, 1], 
            N=[Nt, Nf],
            dims=[dim[0], dim+1], include_on_axis_wavelets=True)
        
        print(list(self.ws_tf.sws_layers.keys()))
        for k, v in self.ws_tf.sws_layers.items():
            if len(k[0]) > 1:
                for p, sp in zip(v.conv_phi.items(), v.conv_psi.items()):
                    cphi = p[1]
                    cpsi = sp[1]
                    print(cphi.output_length, [f.Nx for f in cphi.filters], [f.Nh for f in cphi.filters], [f.ds for f in cphi.filters], [f.ds for f in cpsi.filters])

    def scatteringTransform(self, x: Tensor, batch_size = None, batch_dim = None):
        n_samples = x.shape[batch_dim] if batch_dim != None else 1
        i = 0
        Sbatch = []
        pbar = tqdm(total=n_samples)
        if batch_size == None: batch_size = n_samples
        while i < n_samples:      
            l = min(batch_size, n_samples - i)      
            xb = torch.narrow_copy(x, batch_dim, start=i, length=l) if batch_dim != None else x
            xb = xb.cuda()
            # print(f'MEMORY USAGE: {torch.cuda.memory_allocated()/1024/1024/1024} GB')
            ut, st = self.ws_t.US(xb)
            ut = torch.concat([u.unsqueeze(-1) for u in ut.values()], dim=-1)
            stf = self.ws_tf.scatteringTransform(ut)
            st = torch.concat([s.unsqueeze(-1) for s in st.values()], dim=-1)
            stf = torch.reshape(stf, stf.shape[0:-2] + (-1,))
            Sbatch.append(torch.concat((st, stf), dim=-1).cpu())  
            if batch_dim == None: break 
            pbar.update(l)
            pbar.refresh()
            i += batch_size
            
        pbar.close()
        return torch.concat(Sbatch, dim=batch_dim) if batch_dim != None else Sbatch[0]
    
    def get_total_ds(self) -> List[float]:
        sampl = self.ws_t.samplers
        return [s.d_tot for s in sampl]
        