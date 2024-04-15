import sys
sys.path.append('../python')

import numpy as np
from numpy.fft import fftshift

from sepws.scattering.separable_scattering import SeparableScattering
from sepws.scattering.config import cfg

cfg.set_beta(1.0, 2.5)
cfg.set_alpha(1.0, 2.5)

scat = SeparableScattering([40*10]*2, [8]*2, [[1]*2, [1]*2])

import pprint

for p in scat._calculate_paths():
    pprint.pprint(p, depth = 4, width=200, underscore_numbers=True, compact=True, indent=4)


