import torch
import numpy as np
from dataclasses import dataclass
from torch import Tensor, pi

NUM_ITER = 200

class SEGMM:
    def __init__(self) -> None:
        #first dimension: 0 - signal, 1 - noise
        self.mu: Tensor #mean
        self.sigma: Tensor #std dev
        self.p: Tensor #prior
        
    def fit(self, H : torch.Tensor):
        mu_n = torch.max(H, dim=-1, keepdim=True)[0]
        mu_s = torch.min(H, dim=-1, keepdim=True)[0]
        #note: stack inserts a new dimension in the beginning of the tensor
        self.mu = torch.stack((mu_s, mu_n))
        sigma_n = mu_n - mu_s
        sigma_s = mu_n - mu_s
        self.sigma = torch.stack((sigma_s, sigma_n))
        p_n = torch.ones_like(mu_n)*0.5
        p_s = torch.ones_like(mu_n)*0.5
        self.p = torch.stack((p_s, p_n))   
        
        H = H[None, :]
        
        #TODO: use LL to check for convergence and max it over batch dimensions
        for _ in range(NUM_ITER):            
            z = (H - self.mu)
            w = self._posterior(z)
            s_t = torch.sum(w, -1, keepdim=True)
            self.p = s_t / H.shape[-1]
            self.mu = torch.sum(w*H, -1, keepdim=True) / s_t
            self.sigma = torch.sum(w*z*z, -1, keepdim=True) / s_t
        
        #TODO: go through each batch dimension (dim 1) and ensure that the mu's
        #      are the correct way around
        
        # if self.mu[0, 0].item() > self.mu[1, 0].item():
        #     self.mu = torch.flipud(self.mu)
        #     self.p = torch.flipud(self.p)
        #     self.sigma = torch.flipud(self.sigma)
            
    def _gauss(self, z):
        return self.p / ((2*pi)**0.5) / self.sigma * torch.exp(-0.5 * z * z / self.sigma)
     
    def _posterior(self, z):
        pdf = self._gauss(z)
        return pdf / torch.sum(pdf, 0, keepdim=True)
    
    def signal_probability(self, H: Tensor):
        return self._posterior(H - self.mu)[0]



class SEDetector:
    def __init__(self, Tmed, fs, decimate=None) -> None:
        self.Tmed = Tmed
        self.fs = fs
        self.M = int(np.floor(Tmed*fs/2))*2 + 1
        self.segmm = SEGMM()
        self.decimate = self.M if decimate == None else decimate
        
    def _entropy(self, s: torch.Tensor):
        p = s / torch.sum(s, dim = -2, keepdim=True)        
        return -torch.sum(p * torch.log(p + 1e-12), dim=-2, keepdim=False)
    
    def _mf(self, H: Tensor):
        h = torch.nn.functional.pad(H, (self.M//2, self.M//2), mode='reflect')
        return h.unfold(-1, 
                    self.M, self.decimate).median(-1)[0]        
    
    def signal_probability(self, s: Tensor):
        Hmf = self._mf(self._entropy(s))
        self.segmm.fit(Hmf)
        return self.segmm.signal_probability(Hmf)
        
        