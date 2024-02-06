from phd.scattering import config
config.set_precision('single')

from phd.dataprocessing.esc50 import pre_process

pre_process(44100/3)