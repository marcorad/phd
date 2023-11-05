import sys
import time

sys.path.append("../python")

import scattering.scattering_tf as stf
import scattering.scattering_1d as s1d

approx_support = 3.0

conf1d = s1d.ScatteringFB1DConfig(16, 10.0, 250.0, downsample_mode=s1d.DownsampleMode.MAXIMUM_UNIFORM, approximation_support=approx_support, fstart=15.0)
config = stf.MorletFBConfigTF.from_1D(conf1d, 1, 1, 5.0, approximation_support=approx_support)

print(config.time_config.downsample_map)
print(config.quefrequency_config.downsample_map)


fb = stf.ScatteringFilterBank2D(config)



fb.plot_wavelet(6)
# fb.plot(oversample=8)
# fb.plot_lpf()