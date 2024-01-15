import torch
import numpy as np
from typing import Literal
from dataclasses import dataclass

_TORCH_COMPLEX128= torch.complex128
_TORCH_REAL64 = torch.float64
_TORCH_COMPLEX64= torch.complex64
_TORCH_REAL32 = torch.float32

_NUMPY_COMPLEX128 = np.complex128
_NUMPY_REAL64 = np.float64
_NUMPY_COMPLEX64 = np.complex64
_NUMPY_REAL32 = np.float32

TORCH_COMPLEX = _TORCH_COMPLEX128
TORCH_REAL = _TORCH_REAL64

NUMPY_COMPLEX = _NUMPY_COMPLEX128
NUMPY_REAL = _NUMPY_REAL64

def set_precision(prec: Literal['single', 'double']):
    global TORCH_COMPLEX, TORCH_REAL, NUMPY_COMPLEX, NUMPY_REAL
    if prec == 'single':
        TORCH_COMPLEX = _TORCH_COMPLEX64
        TORCH_REAL = _TORCH_REAL32
        NUMPY_COMPLEX = _NUMPY_COMPLEX64
        NUMPY_REAL = _NUMPY_REAL32
    
    elif prec == 'double':
        TORCH_COMPLEX = _TORCH_COMPLEX128
        TORCH_REAL = _TORCH_REAL64
        NUMPY_COMPLEX = _NUMPY_COMPLEX128
        NUMPY_REAL = _NUMPY_REAL64

@dataclass
class MorletDefinition:
    _alpha_large_Q: float
    beta: float
    _alpha_small_Q: float
    Q_cutoff: float
    k: int

    def alpha(self, Q):
        if Q > self.Q_cutoff:
            return self._alpha_large_Q
        else:
            return self._alpha_small_Q
        

MORLET_DEF_PERFORMANCE =    MorletDefinition(2, 1, 3, 2, 3)
MORLET_DEF_ACCURACY =       MorletDefinition(2, 3, 3, 2, 5)
MORLET_DEF_BALANCED =       MorletDefinition(2, 2, 3, 2, 4)

MORLET_DEFINITION = MORLET_DEF_BALANCED


