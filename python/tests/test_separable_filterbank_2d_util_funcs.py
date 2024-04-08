import sys
sys.path.append('../python')

import numpy as np
from numpy.fft import fftshift

from sepws.scattering.filterbank import scattering_filterbanks, calculate_padding_1d, filterbank_to_tensor, get_Lambda_set, get_wavelet_filter
from sepws.scattering.config import cfg



N = 28 #MNIST size
d = 8
l, r, Nt = calculate_padding_1d(N, d)
print(l, r, Nt, Nt % d, Nt // d - 2)


oversample = 10

fb = scattering_filterbanks([Nt * oversample]*2, [d]*2, [[2]*2, [1]*2],startfreq=None)
cfg.cuda()
filterbank_to_tensor(fb)
print(get_wavelet_filter(fb, 0, 0, 1, 0).dtype)
print(get_wavelet_filter(fb, 0, 0, 1, 0).device)

fb = scattering_filterbanks([Nt * oversample]*2, [d]*2, [[2]*2, [1]*2],startfreq=None)
cfg.cpu()
filterbank_to_tensor(fb)
print(get_wavelet_filter(fb, 0, 0, 1, 0).device)

print(get_Lambda_set(fb, 0, [1, 1]))
print(get_Lambda_set(fb, 1, [4, 4]))


 