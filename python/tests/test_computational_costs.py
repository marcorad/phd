from math import log2

m = 2
N = [256] * m
r = [2] * m

def prod(x):
    n_prod = 1
    for n in x: n_prod *= n
    return n_prod

def calc_K(r):
    m = len(r)
    K = [1]
    for i in range(1, m):
        K.append(K[-1]*r[i])
    return K

def sep_fft_cost(N, r):
    m = len(r)
    K = calc_K(r)
    cost = 0
    n_prod = prod(N)
    for i in range(m):
        cost += n_prod / K[i] * log2 (N[i] / K[i])
    P = 1
    for ni, ri in zip(N, r): P *= ni/ri
    return cost + P * log2(P)

def sep_mul_cost(N, r):
    m = len(r)
    K = calc_K(r)
    cost = 0
    for i in range(m):
        p = 1
        for j in range(m):
            p *= N[j] / K[i]
        cost += p
    return cost

def non_sep_fft_cost(N, r):
    m = len(r)
    n_prod = prod(N)
    P = 1
    for ni, ri in zip(N, r): P *= ni/ri
    return n_prod * log2(n_prod) + P * log2(P)

def non_sep_mul_cost(N, r):
    return prod(N)

print(f'Sep     fft {sep_fft_cost(N, r)} mul {sep_mul_cost(N, r)}')
print(f'Non-sep fft {non_sep_fft_cost(N, r)} mul {non_sep_mul_cost(N, r)}')

import matplotlib.pyplot as plt
import numpy as np

N = [2**i for i in range(4, 16, 4)]
D = np.arange(1, 9)
print(N)
for m in [2, 3]:
    C = []
    for d in D:
        costs = []
        for n in N:
            sc = (sep_fft_cost([n] * m, [d] * m) + 1*sep_mul_cost([n] * m, [d] * m))
            nsc = (non_sep_fft_cost([n] * m, [d] * m) + 1*non_sep_mul_cost([n] * m, [d] * m))
            costs.append(nsc/sc)
        C.append(costs)
       
    C = np.array(C) 
    plt.subplot(2, 1, m-1)
    plt.plot(D, C)
    plt.title(f'{m}D')
    plt.legend(["x".join([str(n) for _ in range(m)]) for n in N])
    
plt.xlabel('Downsample factor (shared across dimensions)')
    
plt.show(block=True)
    
    