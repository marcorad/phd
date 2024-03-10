import skimage as ski
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt


im = ski.data.chelsea()
# im = ski.color.rgb2gray(im)
im = im.astype(np.float32)/255
X = fftshift(np.log(np.abs(fft2(im))))
X = (X - np.min(X))
X /= np.max(X)




plt.subplot(211)
plt.imshow(im)
plt.subplot(212)
i = plt.imshow(X)
# i.norm.autoscale([np.min(X), np.max(X)])
plt.show(block=True)