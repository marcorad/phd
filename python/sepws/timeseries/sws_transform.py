from ..scattering import sep_ws as ws
import numpy as np
import torch
from torch import Tensor
from typing import List, Dict
from math import ceil, floor
from tqdm import tqdm
from ..scattering import config


   

def time_series_sws_tranform(X: List[np.ndarray], sws: ws.SeperableWaveletScattering | ws.JointTFScattering, time_dim: int, flatten = True, func = None) -> List[Tensor]:
    """
    Take numerous observations of a single time series and perform the seperable wavelet scattering transform on each.
    Each time series will be sliced according the SWS length in the time dimension, with padding added and then removed where necessary.
    This means that some time series will have boundary artifacts at these slices, which may affect results. 
    Also useful for longer time series where speed must be increased (shorter FFTs required) or memory must be reduced.

    Args:
        X (List[np.ndarray]): List of time series observations
        sws (ws.SeperableWaveletScattering): SWS transform to apply
        time_dim (int): The time dimension, which must a convolution dimension in sws.
        flatten (bool, optional): Defaults to True. Whether the features should be flattened so that each observtion has shape (n_time, n_feat). 

    Returns:
        List[Tensor]: The SWS tranform of the time series observations
    """ 
    S : List[Tensor] = []
    if time_dim in sws.dims:   
        d_tot = sws.get_total_ds()
        time_dim_index = sws.dims.index(time_dim)
        sws_time_size = sws.N[time_dim_index]
        for x in tqdm(X):
            x_torch = torch.from_numpy(x).type(config.TORCH_REAL)
            xmin, xmax = torch.min(x_torch), torch.max(x_torch)
            x_torch = (x_torch - xmin) / (xmax - xmin)
            n = x_torch.shape[time_dim]
            i = 0
            s_x = []
            #take each time series and slice it to fit with the expected sws length
            while i < n:
                length = min(n - i, sws_time_size)
                x_slice = x_torch.narrow(time_dim, i, length)
                i += length
                n_pad = sws_time_size - x_slice.shape[time_dim]
                if n_pad > 0: #pad the end if necessary
                    x_slice = x_slice.swapaxes(-1, time_dim)
                    x_slice = torch.nn.functional.pad(x_slice, [0, n_pad])
                    x_slice = x_slice.swapaxes(-1, time_dim)
                s: Tensor = sws.scatteringTransform(x_slice).cpu()
                if n_pad > 0:
                    ds_len = ceil(length / d_tot[time_dim_index])
                    s = s.narrow(time_dim, 0, ds_len)
                s_x.append(s)
            S.append(torch.concat(s_x, dim=time_dim)) #join all sliced sections and add to the sws transform list
    else:
        for x in tqdm(X):
            x_torch = torch.from_numpy(x).type(config.TORCH_REAL)
            xmin, xmax = torch.min(x_torch), torch.max(x_torch)
            x_torch = (x_torch - xmin) / (xmax - xmin)
            S.append(sws.scatteringTransform(x_torch).cpu())
            
            
    if func != None: 
        for i, s in enumerate(S): S[i] = func(s)         
    
    if flatten:
        for i, s in enumerate(S):
            s = s.swapaxes(0, time_dim)
            s = s.reshape((s.shape[0], -1))
            S[i] = s     
            
    for i, s in enumerate(S): S[i] = s.numpy()   
                
    return S
                