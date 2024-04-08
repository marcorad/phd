from sepws.scattering import config
config.set_precision('single')

from sepws.dataprocessing.esc50 import pre_process

pre_process(44100/3)