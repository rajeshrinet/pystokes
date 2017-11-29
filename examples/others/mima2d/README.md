## MIMA in pystokes
This is the part of the pystokes library which solves Stokes equation by mollifications of the delta function associated with each particle

## Usage:

```
from __future__ import division
import numpy as np
import pystokes
import matplotlib.pyplot as plt

#parameters
dim, a, Np = 2, 35, 1
Nx, Ny = 128, 128
Lx, Ly = 128, 128

# memory allocation
v = np.zeros(dim*Nx*Ny); F = np.zeros(dim*Np); r = np.zeros(dim*Np); 

# Initialise
r[0], r[1] = Nx/2 , Nx/2 
F[Np:2*Np]=1

mFlow = pystokes.mima2D.Flow(a, Np, Lx, Ly, Nx, Ny);
mFlow.stokesletV(v, r, F, sigma=a/3, NN=10)           # NN is nearest neighbour u connsider for constructing 
                                                      # Gaussian mollifier of standard deviation sigma.                     
```   
