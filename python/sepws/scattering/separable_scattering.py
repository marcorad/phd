from .config import cfg
from .torch_backend import TorchBackend
from .filterbank import scattering_filterbanks, get_Lambda_set, get_wavelet_filter, filterbank_to_tensor, calculate_padding_1d, get_output_downsample_factor, calculate_sigma_psi_w, calculate_sigma_phi_w
from typing import List, Tuple, Dict
from torch import Tensor

class SeparableScattering:
    backend: TorchBackend = TorchBackend()
    def __init__(self, N: List[int], d: List[int], Q: List[List[float]], startfreq: List[float] = None, allow_ds = True) -> None:
        self.pad = []
        self.Npad = []
        self.Ndim = len(N)
        self.d = d
        self.Q = Q
        self.Nlevels = len(Q)
        self.allow_ds = allow_ds
        assert self.Nlevels <= 3, f'Requested {self.Nlevels} scattering levels. A maximum of 3 levels is supported. Higher levels than 3 are typically not useful.'
        assert all([d[i] <= N[i] for i in range(self.Ndim)]), f'All invariance scales {d} must be <= the signal support {N}.'
        for i in reversed(range(self.Ndim)):
            l, r, n = calculate_padding_1d(N[i], d[i])
            self.pad.extend([l, r])
            self.Npad.append(n)
        self.Npad.reverse()
            
        self.fb = scattering_filterbanks(self.Npad, d, Q, startfreq, allow_ds)
        filterbank_to_tensor(self.fb)
        
    def _mul_and_downsample(self, X: Tensor, level: int, input_ds: List[int], lambdas: List[float]):
        mul1d = lambda x, y, dim: self.backend.mul1d(x, y, dim+1) #+1 to account for batch dim
        freqds = lambda x, d, dim: self.backend.freq_downsample1d(x, d, dim+1)
        for dim in range(self.Ndim):
            filter = get_wavelet_filter(self.fb, level, dim, input_ds[dim], lambdas[dim])  
            ds = get_output_downsample_factor(self.fb, level, dim, input_ds[dim], lambdas[dim])
            X = mul1d(X, filter, dim)
            if self.allow_ds: X = freqds(X, ds, dim)
        return X
    
    def _ifft_mul_and_downsample(self, X: Tensor, level: int, input_ds: List[int], lambdas: List[float]):
        mul1d = lambda x, y, dim: self.backend.mul1d(x, y, dim+1) #+1 to account for batch dim
        freqds = lambda x, d, dim: self.backend.freq_downsample1d(x, d, dim+1)
        ifft1d = lambda x, dim: self.backend.ifft1d(x, dim)
        for dim in range(self.Ndim):
            filter = get_wavelet_filter(self.fb, level, dim, input_ds[dim], lambdas[dim])  
            ds = get_output_downsample_factor(self.fb, level, dim, input_ds[dim], lambdas[dim])
            X = mul1d(X, filter, dim)
            if self.allow_ds: X = freqds(X, ds, dim)
            X = ifft1d(X, dim)
        return X
    
    def _ifft_all(self, X: Tensor):
        dims = [1 + dim for dim in range(self.Ndim)]
        return self.backend.ifft(X, dims)
    
    def _fft_all(self, X: Tensor):
        dims = [1 + dim for dim in range(self.Ndim)]
        return self.backend.fft(X, dims)
    
    def _get_compounded_downsample_factor(self, level, current_ds, lambdas):
        if self.allow_ds:
            return [get_output_downsample_factor(self.fb, level, dim, current_ds[dim], lambdas[dim]) * current_ds[dim] for dim in range(self.Ndim)]
        return [1 for _ in range(self.Ndim)]
    
    def _should_prune(self, lambda_filt: List[float], lambda_demod: List[float], level: int):
        bws = []
        for i in range(self.Ndim):
            sigma_psi_w = calculate_sigma_psi_w(self.Q[level][i])
            sigma_phi_w = calculate_sigma_phi_w(self.d[i], self.Q[level][i])
            beta = cfg.get_beta(self.Q[level][i])
            #take into account the linearly spaced filter bandwidths as well
            sigma_psi_w_filt = max(sigma_psi_w * abs(lambda_filt[i]), sigma_phi_w)
            sigma_psi_w_demod = max(sigma_psi_w * abs(lambda_demod[i]), sigma_phi_w)
            # prune only when demodulated filter's bandwith is not at least within beta standard deviations of the current filter
            # note that we use the abs value of the lambdas since lambdas can be positive or negative
            # |----------*----------|
            # |-----ddddd*ddddd-----|
            #            <---->          
            # |----------*--ffxff---|
            #               <->       
            # these intervals must overlap, 
            # * is the centre of the spectrum, d is the significant bandwidth of the demodulated filter (via modulus), and f is the morlet filter under consideration which has a center x
            EPS = 1e-9 #for floating point error
            if abs(lambda_filt[i]) > beta*sigma_psi_w_demod + EPS: return True 
            # if abs(lambda_filt[i]) > abs(lambda_demod[i]): return True 
            bws.append(abs(lambda_filt[i]) - sigma_psi_w_filt > sigma_psi_w_demod)           
        return False
    
    def scattering(self, x: Tensor, normalise = False):
        S, _, _, _ = self._scattering(x, False, False, normalise)        
        return self.backend.stack(S)
    
    def _calculate_paths(self):   #TODO: use for pre-calculating separable filters     
        paths = []                
        lambda_zero = tuple([0 for _ in range(self.Ndim)]) #tuple of zeros for the LPF (phi)
        l0_compounded_ds = [1 for _ in range(self.Ndim)] #no downsampling on the input
        paths.append({
            'level': 0,
            'lambda': (lambda_zero,),
            'phi_ds': l0_compounded_ds
        })        
        #first level
        Lambda_1 = get_Lambda_set(self.fb, 0, [1]*self.Ndim)
        for lambda1 in Lambda_1:  
            l1_compounded_ds = self._get_compounded_downsample_factor(0, l0_compounded_ds, lambda1)
            paths.append({
                'level': 1,
                'lambda': (lambda1,),
                'psi_ds': l0_compounded_ds,
                'phi_ds': l1_compounded_ds,
            })            
            if self.Nlevels == 1: continue            
            #second level
            Lambda_2 = get_Lambda_set(self.fb, 1, l1_compounded_ds)
            for lambda2 in Lambda_2:                
                if self._should_prune(lambda2, lambda1, 1): continue #prune the paths, since downsampling prunes to an inexact extent
                l2_compounded_ds = self._get_compounded_downsample_factor(1, l1_compounded_ds, lambda2)
                paths.append({
                    'level': 2,
                    'lambda': (lambda1, lambda2),
                    'psi_ds': l1_compounded_ds,
                    'phi_ds': l2_compounded_ds
                })              
                if self.Nlevels == 2: continue                
                #third level
                Lambda_3 = get_Lambda_set(self.fb, 2, l2_compounded_ds)
                for lambda3 in Lambda_3:    
                    if self._should_prune(lambda3, lambda2, 2): continue #prune the paths 
                    l3_compounded_ds = self._get_compounded_downsample_factor(2, l2_compounded_ds, lambda3)
                    paths.append({
                    'level': 3,
                    'lambda': (lambda1, lambda2, lambda3),
                    'psi_ds': l2_compounded_ds,
                    'phi_ds': l3_compounded_ds
                })                                                        
        return paths
    
    def _normalise(self, x1: Tensor, xn: Tensor):
        EPS = 1e-10
        return x1 / (xn + EPS)
        
    def _scattering(self, x: Tensor, returnU = False, returnSpath = False, normalise=False):        
        #function aliases for clarity        
        unpad = lambda x: self.backend.unpad(x) if self.allow_ds else x #disable unpadding when DS occurs
        pad = lambda x, s: self.backend.pad(x, s)
        fft = lambda x: self._fft_all(x)
        ifft = lambda x: self._ifft_all(x)        
        mulds = lambda x, level, ids, lambdas: self._mul_and_downsample(x, level, ids, lambdas)
        modulus = lambda x: self.backend.modulus(x)        
        ifftmulds = lambda x, level, ids, lambdas: self._ifft_mul_and_downsample(x, level, ids, lambdas)
        
        #pad the tensor
        x = pad(x, self.pad)
        #get the fft of the input signal across all dimensions
        lambda_zero = tuple([0 for _ in range(self.Ndim)]) #tuple of zeros for the LPF (phi)
        X = fft(x)
        s_0 = unpad(ifft(mulds(X, 0, [1 for _ in range(self.Ndim)], lambda_zero)).real)
        S = [s_0]
        Up = {}
        Sp = {}
        
        l0_compounded_ds = [1 for _ in range(self.Ndim)] #no downsampling on the input
        
        if returnSpath: Sp[lambda_zero] = s_0
        
        #first level
        Lambda_1 = get_Lambda_set(self.fb, 0, [1]*self.Ndim)
        for lambda1 in Lambda_1:            
            u_1 = modulus(ifft(mulds(X, 0, l0_compounded_ds, lambda1)))
            U_1 = fft(u_1)
            l1_compounded_ds = self._get_compounded_downsample_factor(0, l0_compounded_ds, lambda1)
            s_1 = ifft(mulds(U_1, 0, l1_compounded_ds, lambda_zero)).real
            s_1 = unpad(s_1)
            if normalise: s_1 = self._normalise(s_1, s_0)
            S.append(s_1)            
            
            if returnU:     Up[lambda1] = u_1
            if returnSpath: Sp[lambda1] = s_1
            
            if self.Nlevels == 1: continue
            # print(f'{lambda1} ->')
            #second level
            Lambda_2 = get_Lambda_set(self.fb, 1, l1_compounded_ds)
            for lambda2 in Lambda_2:                
                if self._should_prune(lambda2, lambda1, 1): continue #prune the paths, since downsampling prunes to an inexact extent
                # print(f'\t{lambda2}')
                u_2 = modulus(ifft(mulds(U_1, 1, l1_compounded_ds, lambda2)))
                U_2 = fft(u_2)
                l2_compounded_ds = self._get_compounded_downsample_factor(1, l1_compounded_ds, lambda2)
                s_2 = unpad(ifft(mulds(U_2, 1, l2_compounded_ds, lambda_zero)).real)
                if normalise: s_2 = self._normalise(s_2, s_1)
                S.append(s_2)   
                
                if returnU:     Up[(lambda1, lambda2)] = u_2
                if returnSpath: Sp[(lambda1, lambda2)] = s_2
                
                if self.Nlevels == 2: continue
                
                #third level
                Lambda_3 = get_Lambda_set(self.fb, 2, l2_compounded_ds)
                for lambda3 in Lambda_3:    
                    if self._should_prune(lambda3, lambda2, 2): continue #prune the paths
                    u_3 = modulus(ifft(mulds(U_2, 2, l2_compounded_ds, lambda3)))
                    U_3 = fft(u_3)
                    l3_compounded_ds = self._get_compounded_downsample_factor(2, l2_compounded_ds, lambda3)
                    s_3 = unpad(ifft(mulds(U_3, 2, l3_compounded_ds, lambda_zero)).real)
                    if normalise: s_3 = self._normalise(s_3, s_2)
                    S.append(s_3) 
                    
                    if returnU:     Up[(lambda1, lambda2, lambda3)] = u_3
                    if returnSpath: Sp[(lambda1, lambda2, lambda3)] = s_3
                    
        return S, Sp, Up, x
        
        
        
        