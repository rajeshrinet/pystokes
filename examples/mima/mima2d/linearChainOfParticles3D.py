from __future__ import division
import numpy as np
import pystokes
import matplotlib.pyplot as plt

dim = 3;
a, Np = 1, 32
Nx, Ny, Nz = 128, 128, 128
Lx, Ly, Lz = 128, 128, 128
v = np.zeros(dim*Nx*Ny*Ny); 
p = np.zeros(dim*Np); 
S = np.zeros(dim*Np); 
r = np.zeros(dim*Np); 

r[:Np] = 32 + 2*np.arange(32)

p[:Np] = 1
r[Np:2*Np]=Ny/2
r[2*Np:3*Np]=Ny/2
S[:Np] = p[:Np]*p[:Np] -0.5
S[Np:2*Np] = p[:Np]*p[Np:2*Np]

rm = pystokes.mima.Flow(a, Np, Lx, Ly, Lz,  Nx, Ny, Nz);
rm.stressletV(v, r, S)

