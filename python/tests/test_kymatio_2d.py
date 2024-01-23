import numpy as np
from kymatio.torch import Scattering2D
import torch
from time import time

# Set the parameters of the scattering transform.
J = 3
M, N = 32, 32

# Generate a sample signal.
x = np.random.randn(1024, M, N).astype(np.float32)

# Define a Scattering2D object.
S = Scattering2D(J, (M, N))

# Calculate the scattering transform.
t0 = time()
Sx = S.scattering(torch.from_numpy(x))
t1 = time()

print(t1 - t0)

print(Sx.shape)

