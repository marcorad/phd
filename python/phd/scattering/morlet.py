import numpy as np

def sample_gauss(t, sigma):
    return 1.0 / np.sqrt(2.0 * np.pi) / sigma * np.exp(-0.5 * (t / sigma)**2.0)
    
def sample_morlet(t, lambda_, sigma, dir=1.0):
    t_times_lambda = t * lambda_
    g = sample_gauss(t_times_lambda, sigma)
    beta = (sample_gauss(-1.0, 1.0 / sigma) / sample_gauss(0.0, 1.0 / sigma))
    return lambda_ * (np.exp(t_times_lambda*(1j * dir)) - beta) * g

