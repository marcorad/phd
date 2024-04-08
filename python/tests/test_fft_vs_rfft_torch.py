import torch.fft as fft
import torch
import profile

dtype = torch.complex64
device = torch.device('cuda')

N, M = 4096, 4096
x = torch.ones(N, M, dtype=dtype, device=device)

def normal(x, dim):
    for _ in range(100):
        y = fft.fft(x, dim=dim)
    torch.cuda.synchronize() 
        
def inplace(x, dim):
    for _ in range(100):
        fft.fft(x, dim=dim, out=x)  
    torch.cuda.synchronize() 
    
print('DIM 0')
profile.run('normal(x, 0)')
print('DIM 1')
profile.run('normal(x, 1)')

print('INPLACE')
print('DIM 0')
profile.run('inplace(x, 0)')
print('DIM 1')
profile.run('inplace(x, 1)')

#CONCLUSION: INPLACE is NOT faster