# Crowley instability
from __future__ import division
import pystokes
import pyforces
import numpy as np


a, Np1d, Np = 1, 10, 1000         # radius and number of particles
L, dim, eta = 32.2,  3, 1.0/6       # size and dimensionality and the box
latticeShape = 'cube'             # shape of the lattice
v = np.zeros(dim*Np)              # Memory allocation for velocity
r = np.zeros(dim*Np)              # Position vector of the particles
F = np.zeros(dim*Np)              # Forces on the particles
Nb, Nm = 1, 1

rm = pystokes.periodic.Rbm(a, Np, eta, L)   # instantiate the classes
ff = pyforces.forceFields.Forces(Np)

def initialise(r, Np1d, shape):
    ''' this is the module to initialise the particles on the lattice'''
    if shape=='cube':
        for i in range(Np1d):
            for j in range(Np1d):
                for k in range(Np1d):
                    ii = i*Np1d**2 + j*Np1d + k
                    r[ii]      =  3.22*i                  
                    r[ii+Np]   =  3.22*j               
                    r[ii+2*Np] =  3.22*k               
    elif shape=='square':             
        for i in range(Np1d):
            for j in range(Np1d):
                ii = i*Np1d + j
                r[ii]      =  3*i                  
                r[ii+Np]   =  3*j               
        for i in range(Np):
                r[i+2*Np] = L/2
        
    elif shape=='rectangle':             
        for i in range(Np1d):
            for j in range(Np1d):
                ii = i*Np1d + j
                r[ii]      = L/2 + 4*(-Np1d + 2*i)                  
                r[ii+Np]   = L/2 + 4*(-Np1d + 2+ 2*j)               
        for i in range(Np):
                r[i+2*Np] = L/2
    else:
            pass

g=-10
initialise(r, Np1d, latticeShape)
ff.sedimentation(F, g)                   # Sedimentation module of ForceFields 
rm.stokesletV(v, r, F, Nb, Nm)           # StokesletV module of pystokes
cc = (Np-1)/2
v00 = g/(6*np.pi*eta*a) 
phi = 4*np.pi*a**3*Np/(3*L**3)
zickH = 0.2330
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print 'phi', '   ', 'simulation', '   ', 'Zick and Homsy'
print ('%2.3f    %2.3f           %2.3f ' %(phi,  np.mean(v[2*Np:3*Np])/v00,  zickH))
