import numpy as np
from numpy.fft import fft

def sample_gauss(t, sigma):
    return 1.0 / np.sqrt(2.0 * np.pi) / sigma * np.exp(-0.5 * (t / sigma)**2.0)
    
def sample_morlet(t, lambda_, sigma):
    t_times_lambda = t * lambda_
    g = sample_gauss(t_times_lambda, sigma)
    beta = (sample_gauss(-1.0, 1.0 / sigma) / sample_gauss(0.0, 1.0 / sigma))
    return lambda_ * (np.exp(t_times_lambda*(1j)) - beta) * g

def morlet_filter_freq(N, lambda_, sigma):
    hN = N//2
    n = np.arange(-hN, N - hN)
    morlet = sample_morlet(n, lambda_, sigma)
    return np.abs(fft(morlet))

def gauss_filter_freq(N, sigma):
    hN = N//2
    n = np.arange(-hN, N - hN)
    gauss = sample_gauss(n, sigma)
    return np.abs(fft(gauss))

