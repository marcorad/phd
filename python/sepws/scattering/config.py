import torch
from typing import Dict, List, Literal

class Config:
    DEVICE: torch.device = torch.device('cpu')
    COMPLEX_DTYPE: torch.dtype = torch.complex64
    REAL_DTYPE: torch.dtype = torch.float32
    BETA_DEFAULT: float = 2.5
    ALPHA_DEFAULT: float = 2.5
    BETA_LUT: Dict[float, float] = {}
    ALPHA_LUT: Dict[float, float] = {}
    
    def set_beta(self, Q:float, beta: float):
        self.BETA_LUT[Q] = beta
        
    def set_alpha(self, Q: float, alpha: float):
        self.ALPHA_LUT[Q] = alpha
        
    def get_beta(self, Q: float) -> float:
        if Q in self.BETA_LUT.keys(): return self.BETA_LUT[Q]
        return self.BETA_DEFAULT
    
    def get_alpha(self, Q: float) -> float:
        if Q in self.ALPHA_LUT.keys(): return self.ALPHA_LUT[Q]
        return self.ALPHA_DEFAULT
    
    def cuda(self):
        self.DEVICE = torch.device('cuda')
        
    def cpu(self):
        self.DEVICE = torch.device('cpu')
        
    def set_precision(self, prec: Literal['single', 'double']):
        assert(prec in ['single', 'double'])
        if prec == 'single':
            self.REAL_DTYPE = torch.float32
            self.COMPLEX_DTYPE = torch.complex64
        else:
            self.REAL_DTYPE = torch.float64
            self.COMPLEX_DTYPE = torch.complex128
            
cfg = Config()