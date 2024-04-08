from torch import max, abs
from torchaudio.transforms import Resample

import os
import soundfile as sf
import torch

from tqdm import tqdm







# torchaudio.load(uri: Union[BinaryIO, str, PathLike], frame_offset: int = 0, num_frames: int = -1, normalize: bool = True, channels_first: bool = True, format: Optional[str] = None, buffer_size: int = 4096, backend: Optional[str] = None)

class PreProcess:
    def __init__(self, inputdir, outputdir, target_fs, normalise_amp = True) -> None:
        self.files = []
        self.inputdir = inputdir
        self.outputdir = outputdir
        self.normalise_amp = normalise_amp
        self.target_fs = target_fs
        try: 
            os.makedirs(outputdir, exist_ok=True)
            print(f"Created output directory {outputdir}.")
        except Exception as e:
            print(e)
            print(f"Output directory {outputdir} exists.")
        self.target_fs = target_fs
        for entry in os.scandir(inputdir):
            if entry.name.endswith('.wav') and entry.is_file(): 
                self.files.append(entry.name)
                
        
        _, self.orig_fs = sf.read(self.inputdir + "\\" + self.files[0])
                
        self.resample = Resample(self.orig_fs, self.target_fs, dtype=torch.float32)
        
    
        
                
    def process(self):
        for fname in tqdm(self.files): 
            x, _ = sf.read(self.inputdir + "\\" + fname)
            xr = self.resample(torch.Tensor(x))
            xr = xr/max(abs(xr))         
            sf.write(self.outputdir + "\\" + fname, xr, self.target_fs, format='wav')
            
        
        
if __name__ == "__main__":
    pp =PreProcess('D:\Whale Data\AcousticTrends_BlueFinLibrary\casey2017\wav', 'D:\Whale Data\Datasets\Casey2017\\audio', 250)
    # print(pp.files, pp.orig_fs)
    pp.process()
            