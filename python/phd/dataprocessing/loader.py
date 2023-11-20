import torch
import soundfile
import os
import numpy as np
from typing import Literal

from ..scattering.scattering_1d import FB_data_type, TORCH_DATA_TYPE

class DataLoader:
    def __init__(self, dir) -> None:
        self.dir = dir
        self.files = []
        for entry in os.scandir(dir):
            if entry.name.endswith('.wav') and entry.is_file(): 
                self.files.append(entry.name)
                
    def load(self, start, end=None, mode: Literal['2d', '1d'] ='2d') -> torch.Tensor:
        audio = []
        if end == None: end = start+1
        maxlen = 0
        for i in range(start, end):
            x, _ = soundfile.read(self.dir + "\\" + self.files[i])
            x.astype(FB_data_type)
            audio += [x]
            if len(x) > maxlen: maxlen = len(x)  
        
            
        D = np.zeros((len(audio), 1, 1, maxlen), dtype=FB_data_type)
        for i, a in enumerate(audio):
            a = np.pad(a, (0, maxlen - len(a)))
            D[i, 0, 0, :] = a
            
        if mode == '1d':
            D = D[:, :, 0, :] #get rid of the extra 2d dimension
            
        return torch.from_numpy(D)
            
        