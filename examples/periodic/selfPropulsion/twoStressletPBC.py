# Crowley instability
from __future__ import division
import pystokes
import pyforces
import matplotlib.pyplot as plt 
import numpy as np



a,  Np = 1, 2            # radius and number of particles
L, dim = 128,  3                    # size and dimensionality and the box
latticeShape = 'square'             # shape of the lattice
v = np.zeros(dim*Np)                # Memory allocation for velocity
r = np.zeros(dim*Np)                # Position vector of the particles
p = np.zeros(dim*Np)                # Position vector of the particles
S = np.zeros(5*Np)                # Forces on the particles
Nb, Nm = 1, 4



r[:3*Np] = L/2
r[0] = L/2 - L/10
r[1] = L/2 + L/10
p[2*Np:3*Np] = 1

S[:Np]      = p[:Np]*p[:Np] - 1./3
S[Np:2*Np]  = p[Np:2*Np]*p[Np:2*Np] - 1./3
S[2*Np:3*Np]= p[:Np]*p[Np:2*Np]
S[3*Np:4*Np]= p[:Np]*p[2*Np:3*Np]
S[4*Np:5*Np]= p[Np:2*Np]*p[2*Np:3*Np]

dt = 0.1
rm = pystokes.periodic.Rbm(a, Np, 1.0/6, L)   # instantiate the classes

fig = plt.figure()
for tt in range(200):
    v = v*0
    rm.stressletV(v, r, S, Nb, Nm)           # and StokesletV module of pystokes
    r = (r + v*dt)%L
    plt.plot(r[0], r[4], 'o')
    plt.plot(r[1], r[5], '*')
    plt.xlim([0, L])
    plt.ylim([0, L])
    #plt.savefig('Time= %04d.png'%(tt))   # if u want to save the plots instead
    print tt
    plt.pause(0.001)
plt.show()
