import sys

sys.path.append('../python')

from phd.scattering.sep_ws import optimise_T

T = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
fs = 250

for t in T:
    print(optimise_T(t, fs, eps=0.05))